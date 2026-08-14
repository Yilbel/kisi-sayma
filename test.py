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

CIZGI = {"x1": 0, "y1": 630, "x2": 1280, "y2": 630}

HISTEREZIS_PAY = 30
GECIS_ONAY_KARESI = 4
MIN_SAYIM_KUTU_YUKSEKLIGI = 90
KENAR_PAY = 15
TRACK_KAYIP_SURESI = 45


# ============================================================
# BOY KALİBRASYONU
# ============================================================

# Boy kalibrasyonu tam istediğin gibi 1.75 metreye sabitlendi.
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
TAHMIN_HER_N_KAREDE_BIR = 10
MIN_KUTU_BOYUTU = 90

gecis_kayitlari = []
gecis_kilidi = threading.Lock()

# Artık kişilerin sadece boyunu değil, X ve Y ayak koordinatlarını da tutuyoruz
son_kisiler_bilgisi = {}
son_kutu_kilidi = threading.Lock()

track_son_gorulme = {}
isleme_kilidi = threading.Lock()

son_yayin_karesi = None
son_yayin_zamani = 0.0
yayin_kilidi = threading.Lock()

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
    if bant in cocuk: return "Cocuk"
    if bant in genc: return "Genc"
    if bant in yetiskin: return "Yetiskin"
    return "Bilinmiyor"


def cinsiyet_cevir(label):
    label = str(label).lower()
    if "female" in label or "kadin" in label: return "Kadin"
    if "male" in label or "erkek" in label: return "Erkek"
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
# ÇİZGİ KOORDİNAT HESAPLAMA YARDIMCISI
# ============================================================
def get_line_y(x):
    global CIZGI
    x1, y1, x2, y2 = CIZGI["x1"], CIZGI["y1"], CIZGI["x2"], CIZGI["y2"]
    if x2 == x1:
        return y1
    m = (y2 - y1) / (x2 - x1)
    return int(y1 + m * (x - x1))


# ============================================================
# KALİBRASYON (YENİ: Çizgiye En Yakın Kişiyi Bulur)
# ============================================================

def kalibrasyonu_uygula():
    global pixels_per_meter, is_calibrated

    with son_kutu_kilidi:
        if not son_kisiler_bilgisi:
            print("[UYARI] Kalibrasyon için ekranda kimse görünmüyor!")
            return
        
        en_yakin_mesafe = float('inf')
        secilen_h = 0
        secilen_id = None
        
        # Ekrandaki herkes için çizgiye uzaklık hesaplanır
        for tid, bilgi in son_kisiler_bilgisi.items():
            cizgi_y = get_line_y(bilgi["x"])
            mesafe = abs(cizgi_y - bilgi["y"])  # Kişinin ayağı ile çizgi arasındaki Y farkı
            
            # Eğer kişi çizgiye diğerlerinden daha yakınsa, onu seç
            if mesafe < en_yakin_mesafe:
                en_yakin_mesafe = mesafe
                secilen_h = bilgi["h"]
                secilen_id = tid

    if secilen_h <= 0:
        return

    yeni_px_meter = secilen_h / KNOWN_USER_HEIGHT_M

    with kalibrasyon_kilidi:
        pixels_per_meter = yeni_px_meter
        is_calibrated = True

    print("\n" + "=" * 60)
    print("KALİBRASYON TAMAMLANDI")
    print(f"Referans Alınan Kişi ID  : {secilen_id} (Çizgiye uzaklık: {en_yakin_mesafe} px)")
    print(f"Referans Kutu Yüksekliği : {secilen_h} px")
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

def ayak_konumundan_bolge_bul(ayak_x, ayak_y):
    cizgi_y = get_line_y(ayak_x)
    
    ust_sinir = cizgi_y - HISTEREZIS_PAY
    alt_sinir = cizgi_y + HISTEREZIS_PAY
    
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
# TEK KARE İŞLEME
# ============================================================

def kareyi_isle(frame):
    global KARE_SAYAC, CIZGI

    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    annotated_frame = frame.copy()

    with isleme_kilidi:
        KARE_SAYAC += 1
        kare_sayac_simdi = KARE_SAYAC

        x1, y1, x2, y2 = CIZGI["x1"], CIZGI["y1"], CIZGI["x2"], CIZGI["y2"]
        cv2.line(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.line(annotated_frame, (x1, y1 - HISTEREZIS_PAY), (x2, y2 - HISTEREZIS_PAY), (0, 255, 255), 1)
        cv2.line(annotated_frame, (x1, y1 + HISTEREZIS_PAY), (x2, y2 + HISTEREZIS_PAY), (0, 255, 255), 1)

        try:
            results = model.track(
                frame, classes=[0], persist=True, tracker=BYTETRACK_CFG,
                verbose=False, conf=0.35, iou=0.5
            )
        except Exception:
            return annotated_frame

        result = results[0]
        guncel_kisiler = {}

        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.int().cpu().tolist()
            boxes = result.boxes.xyxy.cpu().tolist()

            for track_id, box in zip(ids, boxes):
                bx1, by1, bx2, by2 = [int(v) for v in box]
                bx1 = max(0, min(FRAME_W - 1, bx1))
                by1 = max(0, min(FRAME_H - 1, by1))
                bx2 = max(0, min(FRAME_W - 1, bx2))
                by2 = max(0, min(FRAME_H - 1, by2))

                kutu_w, kutu_h = bx2 - bx1, by2 - by1
                if kutu_w <= 0 or kutu_h <= 0:
                    continue

                ayak_x, ayak_y = int((bx1 + bx2) / 2), by2
                
                track_son_gorulme[track_id] = time.time()
                
                # Kalibrasyon algoritmamız için kişinin tüm konum ve boy bilgilerini ekliyoruz
                guncel_kisiler[track_id] = {"h": kutu_h, "x": ayak_x, "y": ayak_y}

                cv2.circle(annotated_frame, (ayak_x, ayak_y), 6, (255, 0, 255), -1)
                cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

                if track_id not in kisi_bilgisi:
                    deneme_sayisi.setdefault(track_id, 0)
                    if (
                        deneme_sayisi[track_id] < MAX_DENEME
                        and kare_sayac_simdi % TAHMIN_HER_N_KAREDE_BIR == 0
                        and kutu_w >= MIN_KUTU_BOYUTU
                        and kutu_h >= MIN_KUTU_BOYUTU
                    ):
                        yg, cins = yas_cinsiyet_tahmin_et(frame, (bx1, by1, bx2, by2), track_id)
                        deneme_sayisi[track_id] += 1
                        if yg and cins:
                            kisi_bilgisi[track_id] = {"yas_grubu": yg, "cinsiyet": cins}
                        elif deneme_sayisi[track_id] >= MAX_DENEME:
                            kisi_bilgisi[track_id] = {"yas_grubu": "Bilinmiyor", "cinsiyet": "Bilinmiyor"}

                kisi_boyu_m = None
                with kalibrasyon_kilidi:
                    if is_calibrated and pixels_per_meter > 0:
                        kisi_boyu_m = kutu_h / pixels_per_meter

                guvenilir = (
                    kutu_h >= MIN_SAYIM_KUTU_YUKSEKLIGI
                    and bx1 > KENAR_PAY
                    and bx2 < FRAME_W - KENAR_PAY
                    and 0 <= ayak_y < FRAME_H
                )

                if guvenilir:
                    anlik_konum = ayak_konumundan_bolge_bul(ayak_x, ayak_y)
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

                cv2.putText(annotated_frame, etiket, (bx1, max(20, by1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 0), 2)

        with son_kutu_kilidi:
            son_kisiler_bilgisi.clear()
            son_kisiler_bilgisi.update(guncel_kisiler)

        if kare_sayac_simdi % 30 == 0:
            eski_trackleri_temizle()

        with kalibrasyon_kilidi:
            px_text = f"{pixels_per_meter:.0f}px/m" if is_calibrated else "--px/m"
        cv2.putText(annotated_frame, px_text, (20, FRAME_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return annotated_frame


# ============================================================
# FLASK BÖLÜMÜ
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

        with yayin_kilidi:
            son_yayin_karesi = sonuc_data_url
            son_yayin_zamani = time.time()

        return jsonify({"image": sonuc_data_url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/son_kare")
def son_kare():
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


@app.route("/calibrate", methods=["POST"])
def calibrate_via_web():
    kalibrasyonu_uygula()
    return jsonify({"status": "ok"})


@app.route("/set_line", methods=["POST"])
def set_line():
    global CIZGI
    data = request.json
    with isleme_kilidi:
        CIZGI["x1"] = int(data.get("x1", CIZGI["x1"]))
        CIZGI["y1"] = int(data.get("y1", CIZGI["y1"]))
        CIZGI["x2"] = int(data.get("x2", CIZGI["x2"]))
        CIZGI["y2"] = int(data.get("y2", CIZGI["y2"]))
    return jsonify({"status": "ok", "cizgi": CIZGI})


@app.route("/get_line", methods=["GET"])
def get_line():
    return jsonify(CIZGI)


# --- ANA SAYFA (Yayın ve İzleme Ekranı) ---
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
            h1 { color: #00ff88; font-size: 24px; margin-bottom: 10px;}
            img { width: 640px; max-width: 95vw; border: 3px solid #444; border-radius: 8px; background: #000; }
            .btn { background-color: #00ff88; color: #111; padding: 12px 24px; font-size: 16px; font-weight: bold; border: none; border-radius: 5px; cursor: pointer; margin: 8px; }
            .btn:hover { background-color: #00cc6a; }
            .btn.izle { background-color: #3399ff; }
            .btn.izle:hover { background-color: #1177dd; }
            .btn.admin { background-color: #ff4444; color: white; }
            .btn.admin:hover { background-color: #cc0000; }
            .btn:disabled { background-color: #555; color: #999; cursor: not-allowed; }
            .info { margin-top: 15px; font-size: 14px; color: #ccc; }
            .buton-grubu { display: flex; justify-content: center; flex-wrap: wrap; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <h1>Kişi Sayma Sistemi - İzleme Ekranı</h1>

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
            <button class="btn" id="btnKamera" onclick="buAygitinKamerasiniKullan()">Bu Aygıtın Kamerasını Aç</button>
            <button class="btn izle" id="btnIzle" onclick="sunucuyaBagalan()">Sunucuya Bağlan (İzle)</button>
        </div>
        
        <div class="buton-grubu">
            <button class="btn admin" onclick="window.location.href='/admin'">⚙️ Yönetici Paneline Git</button>
        </div>

        <div id="yayinDurumu">Mod: seçilmedi</div>

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
            let aktifMod = null;

            function modlariSifirla() {
                kameraAktif = false;
                izlemeAktif = false;
                btnKamera.disabled = false;
                btnIzle.disabled = false;
            }

            async function buAygitinKamerasiniKullan() {
                if (aktifMod === 'kamera') return;
                modlariSifirla();
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({
                        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
                        audio: false
                    });
                    video.srcObject = stream;
                    kameraAktif = true;
                    aktifMod = 'kamera';
                    btnKamera.disabled = true;
                    yayinDurumuDiv.textContent = "Mod: Yayınlıyor";
                    yayinDurumuDiv.style.color = "#00ff88";
                    goruntuGonder();
                } catch (error) {
                    alert("Kamera hatası: " + error);
                }
            }

            async function goruntuGonder() {
                if (!kameraAktif || aktifMod !== 'kamera') return;
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                
                const dataURL = canvas.toDataURL('image/jpeg', 0.95);
                
                try {
                    const response = await fetch('/process_frame', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({image: dataURL})
                    });
                    const result = await response.json();
                    if (result.image) output.src = result.image;
                } catch (err) {}
                
                setTimeout(goruntuGonder, 40);
            }

            function sunucuyaBagalan() {
                if (aktifMod === 'izle') return;
                modlariSifirla();
                aktifMod = 'izle';
                izlemeAktif = true;
                btnIzle.disabled = true;
                yayinDurumuDiv.textContent = "Mod: İzleniyor...";
                yayinDurumuDiv.style.color = "#3399ff";
                sonKareyiCek();
            }

            async function sonKareyiCek() {
                if (!izlemeAktif || aktifMod !== 'izle') return;
                try {
                    const r = await fetch('/son_kare');
                    const veri = await r.json();
                    if (veri.image) output.src = veri.image;
                } catch (err) {}
                
                setTimeout(sonKareyiCek, 50);
            }

            async function sayaclariGuncelle() {
                try {
                    const r = await fetch('/sayaclar');
                    const veri = await r.json();
                    document.getElementById('girisSayisi').textContent = veri.giris;
                    document.getElementById('cikisSayisi').textContent = veri.cikis;
                } catch (e) {}
            }
            setInterval(sayaclariGuncelle, 1000);
        </script>
    </body>
    </html>
    """

# --- ADMİN PANELI (Çizgi Çizme ve Kalibrasyon) ---
@app.route("/admin")
def admin_panel():
    return """
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <title>Admin Paneli - Ayarlar</title>
        <style>
            body { background: #222; color: white; font-family: Arial; text-align: center; margin: 0; padding: 20px; }
            h1 { color: #ff9900; }
            .canvas-container { position: relative; display: inline-block; border: 3px solid #555; border-radius: 8px; overflow: hidden; background: #000; }
            #bg-img { display: block; width: 640px; height: 360px; pointer-events: none; }
            #draw-canvas { position: absolute; top: 0; left: 0; width: 640px; height: 360px; cursor: crosshair; }
            
            .btn { background-color: #00ff88; color: #111; padding: 12px 24px; font-size: 16px; font-weight: bold; border: none; border-radius: 5px; cursor: pointer; margin: 10px; }
            .btn:hover { background-color: #00cc6a; }
            .btn.kalibre { background-color: #ff9900; }
            .btn.kalibre:hover { background-color: #cc7a00; }
            .btn.geri { background-color: #3399ff; color: white; }
            .btn.geri:hover { background-color: #1177dd; }
            .info { margin-top: 15px; font-size: 14px; color: #ccc; }
        </style>
    </head>
    <body>
        <h1>⚙️ Yönetici Kontrol Paneli</h1>
        
        <div class="info">
            <p>1. Görüntüye <b>basılı tutup sürükleyerek</b> sayım yapmak istediğiniz çizgiyi baştan sona çizin.</p>
            <p>2. Boy kalibrasyonu yaparken kamera karşısına geçin, çizgiye en yakın olan kişinin boyu <b>1.75 m</b> olarak hesaplanacaktır.</p>
        </div>

        <div class="canvas-container">
            <img id="bg-img" alt="Canlı yayın bekleniyor...">
            <canvas id="draw-canvas" width="1280" height="720"></canvas>
        </div>

        <div>
            <button class="btn" onclick="cizgiyiKaydet()">✅ Yeni Çizgiyi Kaydet</button>
            <button class="btn kalibre" onclick="kalibrasyonYap()">📏 Çizgideki Kişi İçin Boy Kalibrasyonu Yap (1.75m)</button>
        </div>
        
        <div style="margin-top: 20px;">
            <button class="btn geri" onclick="window.location.href='/'">⬅️ Ana Sayfaya Dön</button>
        </div>
        
        <div id="durumMesaji" style="color:#00ff88; font-weight:bold; margin-top:10px;"></div>
        <div id="kalibrasyonDurumu" style="margin-top: 10px; color: #aaa;"></div>

        <script>
            const bgImg = document.getElementById('bg-img');
            const canvas = document.getElementById('draw-canvas');
            const ctx = canvas.getContext('2d');
            const durumMesaji = document.getElementById('durumMesaji');
            
            let isDrawing = false;
            let currentLine = {x1: 0, y1: 630, x2: 1280, y2: 630};
            let tempLine = null;

            async function mevcutCizgiyiGetir() {
                try {
                    const r = await fetch('/get_line');
                    currentLine = await r.json();
                    cizimiGuncelle();
                } catch(e) {}
            }
            mevcutCizgiyiGetir();

            async function canliYayiniGuncelle() {
                try {
                    const r = await fetch('/son_kare');
                    const veri = await r.json();
                    if (veri.image) bgImg.src = veri.image;
                } catch (e) {}
                
                setTimeout(canliYayiniGuncelle, 50);
            }
            canliYayiniGuncelle();

            function getMousePos(evt) {
                const rect = canvas.getBoundingClientRect();
                const scaleX = canvas.width / rect.width;
                const scaleY = canvas.height / rect.height;
                return {
                    x: (evt.clientX - rect.left) * scaleX,
                    y: (evt.clientY - rect.top) * scaleY
                };
            }

            canvas.addEventListener('mousedown', (e) => {
                isDrawing = true;
                const pos = getMousePos(e);
                tempLine = { x1: pos.x, y1: pos.y, x2: pos.x, y2: pos.y };
            });

            canvas.addEventListener('mousemove', (e) => {
                if (!isDrawing) return;
                const pos = getMousePos(e);
                tempLine.x2 = pos.x;
                tempLine.y2 = pos.y;
                cizimiGuncelle();
            });

            canvas.addEventListener('mouseup', () => {
                isDrawing = false;
                if(tempLine) currentLine = { ...tempLine };
            });
            
            canvas.addEventListener('mouseleave', () => {
                if (isDrawing) {
                    isDrawing = false;
                    if(tempLine) currentLine = { ...tempLine };
                }
            });

            function cizimiGuncelle() {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                const lineToDraw = isDrawing ? tempLine : currentLine;
                
                if (lineToDraw) {
                    ctx.beginPath();
                    ctx.moveTo(lineToDraw.x1, lineToDraw.y1);
                    ctx.lineTo(lineToDraw.x2, lineToDraw.y2);
                    ctx.strokeStyle = "#FF0000";
                    ctx.lineWidth = 5;
                    ctx.stroke();
                }
            }

            async function cizgiyiKaydet() {
                try {
                    const r = await fetch('/set_line', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            x1: Math.round(currentLine.x1),
                            y1: Math.round(currentLine.y1),
                            x2: Math.round(currentLine.x2),
                            y2: Math.round(currentLine.y2)
                        })
                    });
                    const res = await r.json();
                    if(res.status === 'ok') {
                        durumMesaji.textContent = "Çizgi başarıyla güncellendi!";
                        setTimeout(() => durumMesaji.textContent = "", 3000);
                    }
                } catch(e) {
                    alert("Çizgi kaydedilemedi!");
                }
            }

            async function kalibrasyonYap() {
                try {
                    await fetch('/calibrate', { method: 'POST' });
                    durumMesaji.textContent = "Çizgideki kişiye göre kalibrasyon yapıldı.";
                    setTimeout(() => durumMesaji.textContent = "", 3000);
                } catch(e) {
                    alert("Kalibrasyon yapılamadı!");
                }
            }

            async function kalibrasyonDurumunuGuncelle() {
                try {
                    const r = await fetch('/kalibrasyon_durumu');
                    const veri = await r.json();
                    const div = document.getElementById('kalibrasyonDurumu');
                    if (veri.kalibre) {
                        div.textContent = "Güncel Kalibrasyon: " + veri.px_metre + " px/m";
                        div.style.color = "#00ff88";
                    } else {
                        div.textContent = "Kalibrasyon: Henüz yapılmadı";
                        div.style.color = "#aaa";
                    }
                } catch (e) {}
            }
            setInterval(kalibrasyonDurumunuGuncelle, 2000);
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
