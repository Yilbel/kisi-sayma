import os
import time
import base64
import threading
from datetime import datetime

import cv2
import numpy as np

from flask import Flask, Response, request, jsonify
from ultralytics import YOLO
from PIL import Image
from transformers import pipeline


# ============================================================
# GENEL AYARLAR
# ============================================================

PROGRAM_ADI = "Kisi Sayma ve Boy Olcum Sistemi"

FRAME_W = 1280
FRAME_H = 720

FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000


# ============================================================
# KAPI / ÇİZGİ SAYIM AYARLARI
# ============================================================

CIZGI_Y = 630
HISTEREZIS_PAY = 30
GECIS_ONAY_KARESI = 4
MIN_SAYIM_KUTU_YUKSEKLIGI = 90
KENAR_PAY = 15
TRACK_KAYIP_SURESI = 45


# ============================================================
# BOY KALİBRASYONU
# ============================================================

KNOWN_USER_HEIGHT_M = 1.75

pixels_per_meter = 100.0
is_calibrated = False

kalibrasyon_kilidi = threading.Lock()


# ============================================================
# BYTE TRACK AYARI
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BYTETRACK_CFG = os.path.join(BASE_DIR, "bytetrack_custom.yaml")
if not os.path.isfile(BYTETRACK_CFG):
    BYTETRACK_CFG = "bytetrack.yaml"


# ============================================================
# GLOBAL DURUMLAR
# ============================================================

giris_sayisi = 0
cikis_sayisi = 0

KARE_SAYAC = 0

durumlar = {}
gecis_adaylari = {}
kisi_bilgisi = {}
deneme_sayisi = {}

MAX_DENEME = 10
TAHMIN_HER_N_KAREDE_BIR = 5
MIN_KUTU_BOYUTU = 90

gecis_kayitlari = []
gecis_kilidi = threading.Lock()

son_kutu_yukseklikleri = {}
son_kutu_kilidi = threading.Lock()

track_son_gorulme = {}

# Aynı anda tek bir kare işlensin diye (paylaşılan global durumları
# birden fazla istek aynı anda değiştirmesin).
isleme_kilidi = threading.Lock()

# ------------------------------------------------------------
# CANLI YAYIN DEPOSU
# ------------------------------------------------------------
# "Bu Aygıtın Kamerasını Kullan" ile bağlanan cihaz (örn. akıllı tahta)
# her işlenmiş kareyi buraya yazar. "Sunucuya Bağlan" ile bağlanan
# diğer cihazlar (örn. telefon) sadece burayı okuyup ekranda gösterir,
# kendi kameralarını kullanmazlar.
son_yayin_karesi = None          # data:image/jpeg;base64,... formatında
son_yayin_zamani = 0.0           # bu karenin geldiği zaman (time.time())
yayin_kilidi = threading.Lock()

# Yayın "canlı" sayılsın diye: bu süreden uzun süredir yeni kare
# gelmediyse izleyiciye "yayın yok" bilgisi verilir.
YAYIN_ZAMAN_ASIMI = 5.0


# ============================================================
# YAŞ / CİNSİYET MODELLERİ
# ============================================================

print()
print("=" * 60)
print("YAŞ / CİNSİYET MODELLERİ YÜKLENİYOR")
print("=" * 60)

try:
    age_pipe = pipeline(
        "image-classification",
        model="dima806/fairface_age_image_detection"
    )
    gender_pipe = pipeline(
        "image-classification",
        model="dima806/fairface_gender_image_detection"
    )
    YAS_CINSIYET_AKTIF = True
    print("[OK] Yaş ve cinsiyet modelleri yüklendi.")
except Exception as e:
    print("[UYARI] Yaş/cinsiyet modelleri yüklenemedi:", e)
    age_pipe = None
    gender_pipe = None
    YAS_CINSIYET_AKTIF = False


def yas_bandini_gruba_cevir(bant):
    cocuk = {"0-2", "3-9"}
    genc = {"10-19", "20-29"}
    yetiskin = {"30-39", "40-49", "50-59", "60-69", "70+"}
    if bant in cocuk:
        return "Cocuk"
    if bant in genc:
        return "Genc"
    if bant in yetiskin:
        return "Yetiskin"
    return "Bilinmiyor"


def cinsiyet_cevir(label):
    label = str(label).lower()
    if "female" in label or "kadin" in label:
        return "Kadin"
    if "male" in label or "erkek" in label:
        return "Erkek"
    return "Bilinmiyor"


# ============================================================
# YOLO MODELİ
# ============================================================

MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
try:
    if os.path.isfile(MODEL_PATH):
        model = YOLO(MODEL_PATH)
    else:
        model = YOLO("yolov8n.pt")
    print("[OK] YOLO modeli hazır.")
except Exception as e:
    print("[HATA] YOLO modeli yüklenemedi:", e)
    raise


# ============================================================
# YÜZ / KIRPMA & TAHMİN
# ============================================================

def yuz_bul(kutu_goruntu):
    if kutu_goruntu is None or kutu_goruntu.size == 0:
        return None
    h, w = kutu_goruntu.shape[:2]
    if h < 40 or w < 40:
        return None
    return kutu_goruntu


def yas_cinsiyet_tahmin_et(frame, box, track_id):
    if not YAS_CINSIYET_AKTIF:
        return None, None
    try:
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return None, None

        yukseklik = y2 - y1
        ust_y2 = y1 + int(yukseklik * 0.55)
        kirpinti = frame[y1:ust_y2, x1:x2]
        yuz = yuz_bul(kirpinti)
        if yuz is None:
            return None, None

        rgb = cv2.cvtColor(yuz, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        age_result = age_pipe(pil_img)[0]
        gender_result = gender_pipe(pil_img)[0]

        yas_grubu = yas_bandini_gruba_cevir(age_result["label"])
        cinsiyet = cinsiyet_cevir(gender_result["label"])
        return yas_grubu, cinsiyet
    except Exception:
        return None, None


# ============================================================
# KALİBRASYON
# ============================================================

def kalibrasyonu_uygula():
    global pixels_per_meter, is_calibrated

    with son_kutu_kilidi:
        if not son_kutu_yukseklikleri:
            print("[UYARI] Kalibrasyon için ekranda kimse görünmüyor!")
            return
        track_id, yukseklik_px = max(
            son_kutu_yukseklikleri.items(), key=lambda item: item[1]
        )

    if yukseklik_px <= 0:
        return

    yeni_px_meter = yukseklik_px / KNOWN_USER_HEIGHT_M

    with kalibrasyon_kilidi:
        pixels_per_meter = yeni_px_meter
        is_calibrated = True

    print("\n" + "=" * 60)
    print("KALİBRASYON TAMAMLANDI")
    print(f"Referans Kutu Yüksekliği : {yukseklik_px} px")
    print(f"Gerçek Boy               : {KNOWN_USER_HEIGHT_M:.2f} m")
    print(f"Hesaplanan Piksel/Metre  : {pixels_per_meter:.2f}")
    print("=" * 60 + "\n")


def klavye_dinleyici():
    while True:
        try:
            girdi = input().strip().lower()
            if girdi == "c":
                kalibrasyonu_uygula()
        except EOFError:
            break


# ============================================================
# YÖN VE GEÇİŞ İŞLEMLERİ
# ============================================================

def ayak_konumundan_bolge_bul(ayak_y):
    ust_sinir = CIZGI_Y - HISTEREZIS_PAY
    alt_sinir = CIZGI_Y + HISTEREZIS_PAY
    if ayak_y < ust_sinir:
        return "yukari"
    if ayak_y > alt_sinir:
        return "asagi"
    return None


def gecisi_kaydet(track_id, yon, boy_m):
    global giris_sayisi, cikis_sayisi

    bilgi = kisi_bilgisi.get(
        track_id, {"yas_grubu": "Bilinmiyor", "cinsiyet": "Bilinmiyor"}
    )

    kayit = {
        "zaman": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "yas_grubu": bilgi.get("yas_grubu", "Bilinmiyor"),
        "cinsiyet": bilgi.get("cinsiyet", "Bilinmiyor"),
        "boy_m": round(boy_m, 2) if boy_m is not None else None,
    }

    if yon == "Giris":
        giris_sayisi += 1
        kayit["tip"] = "Giris"
        print(f"\n******** GİRİŞ #{giris_sayisi} ********")
    elif yon == "Cikis":
        cikis_sayisi += 1
        kayit["tip"] = "Cikis"
        print(f"\n******** ÇIKIŞ #{cikis_sayisi} ********")
    else:
        return

    print(f"ID        : {track_id}")
    print(f"Yaş grubu : {kayit['yas_grubu']}")
    print(f"Cinsiyet  : {kayit['cinsiyet']}")
    print(f"Boy       : {kayit['boy_m']} m")
    print("*******************************")

    with gecis_kilidi:
        gecis_kayitlari.append(kayit)


def eski_trackleri_temizle():
    simdi = time.time()
    silinecekler = [
        tid for tid, zmn in track_son_gorulme.items()
        if simdi - zmn > TRACK_KAYIP_SURESI
    ]
    for tid in silinecekler:
        track_son_gorulme.pop(tid, None)
        durumlar.pop(tid, None)
        gecis_adaylari.pop(tid, None)
        kisi_bilgisi.pop(tid, None)
        deneme_sayisi.pop(tid, None)


# ============================================================
# TEK KARE İŞLEME (tarayıcıdan gelen her kare burada işlenir)
# ============================================================

def kareyi_isle(frame):
    global KARE_SAYAC

    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    annotated_frame = frame.copy()

    with isleme_kilidi:
        KARE_SAYAC += 1
        kare_sayac_simdi = KARE_SAYAC

        cv2.line(annotated_frame, (0, CIZGI_Y), (FRAME_W, CIZGI_Y), (0, 0, 255), 3)
        cv2.line(annotated_frame, (0, CIZGI_Y - HISTEREZIS_PAY), (FRAME_W, CIZGI_Y - HISTEREZIS_PAY), (0, 255, 255), 1)
        cv2.line(annotated_frame, (0, CIZGI_Y + HISTEREZIS_PAY), (FRAME_W, CIZGI_Y + HISTEREZIS_PAY), (0, 255, 255), 1)

        try:
            results = model.track(
                frame, classes=[0], persist=True, tracker=BYTETRACK_CFG,
                verbose=False, conf=0.35, iou=0.5
            )
        except Exception:
            return annotated_frame

        result = results[0]
        guncel_yukseklikler = {}

        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.int().cpu().tolist()
            boxes = result.boxes.xyxy.cpu().tolist()

            for track_id, box in zip(ids, boxes):
                x1, y1, x2, y2 = [int(v) for v in box]
                x1 = max(0, min(FRAME_W - 1, x1))
                y1 = max(0, min(FRAME_H - 1, y1))
                x2 = max(0, min(FRAME_W - 1, x2))
                y2 = max(0, min(FRAME_H - 1, y2))

                kutu_w, kutu_h = x2 - x1, y2 - y1
                if kutu_w <= 0 or kutu_h <= 0:
                    continue

                track_son_gorulme[track_id] = time.time()
                guncel_yukseklikler[track_id] = kutu_h

                ayak_x, ayak_y = int((x1 + x2) / 2), y2

                cv2.circle(annotated_frame, (ayak_x, ayak_y), 6, (255, 0, 255), -1)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Yaş / Cinsiyet Tahmini
                if track_id not in kisi_bilgisi:
                    deneme_sayisi.setdefault(track_id, 0)
                    if (
                        deneme_sayisi[track_id] < MAX_DENEME
                        and kare_sayac_simdi % TAHMIN_HER_N_KAREDE_BIR == 0
                        and kutu_w >= MIN_KUTU_BOYUTU
                        and kutu_h >= MIN_KUTU_BOYUTU
                    ):
                        yg, cins = yas_cinsiyet_tahmin_et(frame, (x1, y1, x2, y2), track_id)
                        deneme_sayisi[track_id] += 1
                        if yg and cins:
                            kisi_bilgisi[track_id] = {"yas_grubu": yg, "cinsiyet": cins}
                        elif deneme_sayisi[track_id] >= MAX_DENEME:
                            kisi_bilgisi[track_id] = {"yas_grubu": "Bilinmiyor", "cinsiyet": "Bilinmiyor"}

                # Boy Hesaplama
                kisi_boyu_m = None
                with kalibrasyon_kilidi:
                    if is_calibrated and pixels_per_meter > 0:
                        kisi_boyu_m = kutu_h / pixels_per_meter

                # Sayım Kontrolü
                guvenilir = (
                    kutu_h >= MIN_SAYIM_KUTU_YUKSEKLIGI
                    and x1 > KENAR_PAY
                    and x2 < FRAME_W - KENAR_PAY
                    and 0 <= ayak_y < FRAME_H
                )

                if guvenilir:
                    anlik_konum = ayak_konumundan_bolge_bul(ayak_y)
                    onceki_durum = durumlar.get(track_id)

                    if onceki_durum is None:
                        if anlik_konum is not None:
                            durumlar[track_id] = anlik_konum
                    elif anlik_konum is not None and anlik_konum != onceki_durum:
                        aday = gecis_adaylari.get(track_id)
                        if aday is not None and aday["yon"] == anlik_konum:
                            aday["sayac"] += 1
                        else:
                            aday = {"yon": anlik_konum, "sayac": 1}
                        gecis_adaylari[track_id] = aday

                        if aday["sayac"] >= GECIS_ONAY_KARESI:
                            yeni_durum = anlik_konum
                            if onceki_durum == "yukari" and yeni_durum == "asagi":
                                gecisi_kaydet(track_id, "Giris", kisi_boyu_m)
                            elif onceki_durum == "asagi" and yeni_durum == "yukari":
                                gecisi_kaydet(track_id, "Cikis", kisi_boyu_m)

                            durumlar[track_id] = yeni_durum
                            gecis_adaylari.pop(track_id, None)
                    else:
                        gecis_adaylari.pop(track_id, None)

                etiket = f"ID: {track_id}"
                if track_id in kisi_bilgisi:
                    b = kisi_bilgisi[track_id]
                    etiket += f" | {b['cinsiyet']} | {b['yas_grubu']}"
                if kisi_boyu_m is not None:
                    etiket += f" | {kisi_boyu_m:.2f}m"

                cv2.putText(annotated_frame, etiket, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 0), 2)

        with son_kutu_kilidi:
            son_kutu_yukseklikleri.clear()
            son_kutu_yukseklikleri.update(guncel_yukseklikler)

        if kare_sayac_simdi % 30 == 0:
            eski_trackleri_temizle()

        with kalibrasyon_kilidi:
            px_text = f"{pixels_per_meter:.0f}px/m" if is_calibrated else "--px/m"
        cv2.putText(annotated_frame, px_text, (20, FRAME_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return annotated_frame


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


@app.route("/process_frame", methods=["POST"])
def process_frame():
    global son_yayin_karesi, son_yayin_zamani
    try:
        data = request.json.get("image")
        if not data:
            return jsonify({"error": "Görüntü yok"}), 400

        encoded_data = data.split(",")[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Geçersiz görüntü"}), 400

        annotated_frame = kareyi_isle(frame)

        _, buffer = cv2.imencode(".jpg", annotated_frame)
        jpg_as_text = base64.b64encode(buffer).decode("utf-8")
        sonuc_data_url = f"data:image/jpeg;base64,{jpg_as_text}"

        # İşlenen bu kareyi "canlı yayın" olarak sunucuda sakla, böylece
        # "Sunucuya Bağlan" ile bağlanan diğer cihazlar (kendi kameralarını
        # açmadan) bu kareyi çekip görebilir.
        with yayin_kilidi:
            son_yayin_karesi = sonuc_data_url
            son_yayin_zamani = time.time()

        return jsonify({"image": sonuc_data_url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/son_kare")
def son_kare():
    """
    Kamera göndermeyen, sadece izleyen cihazlar (örn. telefon) bu
    endpoint'i periyodik olarak çağırarak sunucudaki en son işlenmiş
    kareyi alır. Kendi kamerasını açmaz.
    """
    with yayin_kilidi:
        kare = son_yayin_karesi
        zaman = son_yayin_zamani

    if kare is None:
        return jsonify({"yayin_var": False, "image": None})

    yayin_aktif = (time.time() - zaman) <= YAYIN_ZAMAN_ASIMI
    return jsonify({"yayin_var": yayin_aktif, "image": kare})


@app.route("/kalibrasyon_durumu")
def kalibrasyon_durumu():
    with kalibrasyon_kilidi:
        return jsonify({
            "kalibre": is_calibrated,
            "px_metre": round(pixels_per_meter, 1) if is_calibrated else None,
        })


@app.route("/sayaclar")
def sayaclar():
    return jsonify({"giris": giris_sayisi, "cikis": cikis_sayisi})


@app.route("/")
def anasayfa():
    return """
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <title>Kişi Sayma ve Boy Ölçüm Sistemi</title>
        <style>
            body { background: #111; color: white; font-family: Arial; text-align: center; margin: 0; padding: 20px; }
            h1 { color: #00ff88; }
            img { width: 640px; max-width: 95vw; border: 3px solid #444; border-radius: 8px; background: #000; }
            .btn { background-color: #00ff88; color: #111; padding: 12px 24px; font-size: 16px; font-weight: bold; border: none; border-radius: 5px; cursor: pointer; margin: 8px; }
            .btn:hover { background-color: #00cc6a; }
            .btn.izle { background-color: #3399ff; }
            .btn.izle:hover { background-color: #1177dd; }
            .btn:disabled { background-color: #555; color: #999; cursor: not-allowed; }
            .info { margin-top: 15px; font-size: 14px; color: #ccc; }
            #kalibrasyonDurumu, #yayinDurumu { margin-top: 8px; font-size: 14px; color: #aaa; }
            .buton-grubu { display: flex; justify-content: center; flex-wrap: wrap; }
        </style>
    </head>
    <body>
        <h1>Kişi Sayma ve Boy Ölçüm Sistemi</h1>

        <div id="sayacCubugu" style="font-size: 28px; font-weight: bold; margin-bottom: 10px;">
            <span style="color:#00ff88;">GİRİŞ: <span id="girisSayisi">0</span></span>
            &nbsp;&nbsp;&nbsp;&nbsp;
            <span style="color:#ff4444;">ÇIKIŞ: <span id="cikisSayisi">0</span></span>
        </div>

        <div>
            <video id="video" autoplay playsinline style="display:none;"></video>
            <canvas id="canvas" width="1280" height="720" style="display:none;"></canvas>
            <img id="output" alt="Başlamak için aşağıdan bir mod seçin">
        </div>

        <div class="buton-grubu">
            <button class="btn" id="btnKamera" onclick="buAygitinKamerasiniKullan()">Bu Aygıtın Kamerasını Kullan</button>
            <button class="btn izle" id="btnIzle" onclick="sunucuyaBagalan()">Sunucuya Bağlan</button>
        </div>

        <div id="yayinDurumu">Mod: seçilmedi</div>
        <div id="kalibrasyonDurumu">Kalibrasyon: kontrol ediliyor...</div>

        <div class="info">
            <p>Kalibrasyon için, kişi kamera karşısındayken PC'nin çalıştığı terminale <b>c</b> yazıp Enter'a basın.</p>
            <p><b>Bu Aygıtın Kamerasını Kullan:</b> bu cihazın kamerasını açar ve görüntüyü işlenmek üzere sunucuya gönderir.<br>
               <b>Sunucuya Bağlan:</b> kamera açmaz, sadece o an sunucuya gönderilen (başka bir cihazın yayınladığı) işlenmiş görüntüyü izler.</p>
        </div>

        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const output = document.getElementById('output');
            const ctx = canvas.getContext('2d');
            const btnKamera = document.getElementById('btnKamera');
            const btnIzle = document.getElementById('btnIzle');
            const yayinDurumuDiv = document.getElementById('yayinDurumu');

            let kameraAktif = false;
            let izlemeAktif = false;
            let aktifMod = null; // 'kamera' | 'izle' | null

            function modlariSifirla() {
                kameraAktif = false;
                izlemeAktif = false;
                btnKamera.disabled = false;
                btnIzle.disabled = false;
            }

            // ---- MOD 1: Bu aygıtın kamerasını kullan (yayıncı) ----
            async function buAygitinKamerasiniKullan() {
                if (aktifMod === 'kamera') return;
                modlariSifirla();
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({
                        video: {
                            facingMode: "environment",
                            width: { ideal: 1280 },
                            height: { ideal: 720 }
                        },
                        audio: false
                    });
                    video.srcObject = stream;
                    kameraAktif = true;
                    aktifMod = 'kamera';
                    btnKamera.disabled = true;
                    yayinDurumuDiv.textContent = "Mod: Bu aygıtın kamerası kullanılıyor (yayınlıyor)";
                    yayinDurumuDiv.style.color = "#00ff88";
                    goruntuGonder();
                } catch (error) {
                    alert("Kamera izni alınamadı veya cihazda kamera bulunamadı: " + error);
                }
            }

            async function goruntuGonder() {
                if (!kameraAktif || aktifMod !== 'kamera') return;

                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const dataURL = canvas.toDataURL('image/jpeg', 0.85);

                try {
                    const response = await fetch('/process_frame', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({image: dataURL})
                    });
                    const result = await response.json();
                    if (result.image) {
                        output.src = result.image;
                    }
                } catch (err) {
                    console.error("Sunucu iletişim hatası:", err);
                }

                setTimeout(goruntuGonder, 120);
            }

            // ---- MOD 2: Sunucuya bağlan (izleyici) ----
            function sunucuyaBagalan() {
                if (aktifMod === 'izle') return;
                modlariSifirla();
                aktifMod = 'izle';
                izlemeAktif = true;
                btnIzle.disabled = true;
                yayinDurumuDiv.textContent = "Mod: Sunucudaki yayın izleniyor...";
                yayinDurumuDiv.style.color = "#3399ff";
                sonKareyiCek();
            }

            async function sonKareyiCek() {
                if (!izlemeAktif || aktifMod !== 'izle') return;

                try {
                    const r = await fetch('/son_kare');
                    const veri = await r.json();
                    if (veri.image) {
                        output.src = veri.image;
                    }
                    if (veri.yayin_var) {
                        yayinDurumuDiv.textContent = "Mod: Sunucudaki yayın izleniyor";
                        yayinDurumuDiv.style.color = "#3399ff";
                    } else {
                        yayinDurumuDiv.textContent = "Mod: İzleniyor, ancak şu an aktif bir yayın yok";
                        yayinDurumuDiv.style.color = "#ff9900";
                    }
                } catch (err) {
                    console.error("Sunucu iletişim hatası:", err);
                }

                setTimeout(sonKareyiCek, 300);
            }

            async function kalibrasyonDurumunuGuncelle() {
                try {
                    const r = await fetch('/kalibrasyon_durumu');
                    const veri = await r.json();
                    const div = document.getElementById('kalibrasyonDurumu');
                    if (veri.kalibre) {
                        div.textContent = "Kalibrasyon: Yapıldı (" + veri.px_metre + " px/m)";
                        div.style.color = "#00ff88";
                    } else {
                        div.textContent = "Kalibrasyon: Yapılmadı";
                        div.style.color = "#aaa";
                    }
                } catch (e) {}
            }

            async function sayaclariGuncelle() {
                try {
                    const r = await fetch('/sayaclar');
                    const veri = await r.json();
                    document.getElementById('girisSayisi').textContent = veri.giris;
                    document.getElementById('cikisSayisi').textContent = veri.cikis;
                } catch (e) {}
            }

            kalibrasyonDurumunuGuncelle();
            setInterval(kalibrasyonDurumunuGuncelle, 2000);
            sayaclariGuncelle();
            setInterval(sayaclariGuncelle, 1000);
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    threading.Thread(target=klavye_dinleyici, daemon=True).start()

    try:
        app.run(host=FLASK_HOST, port=FLASK_PORT, threaded=True, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        pass