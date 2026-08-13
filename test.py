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
# Çizgi artık sabit yatay değil, admin panelinden iki nokta (p1, p2)
# olarak tanımlanıyor. HISTEREZIS_PAY artık çizgiye dik mesafe olarak,
# KENAR_PAY ise çizgi boyunca (uçlara yakın "ölü alan") olarak
# kullanılıyor -- yani ikisi de çizgiyle birlikte hareket ediyor.

GECIS_ONAY_KARESI = 4
MIN_SAYIM_KUTU_YUKSEKLIGI = 90
KENAR_PAY = 15
TRACK_KAYIP_SURESI = 45

# Varsayılan çizgi: önceki çalışan sürümdeki CIZGI_Y=630 ile aynı konum
cizgi_p1 = [0.0, 630.0]
cizgi_p2 = [float(FRAME_W), 630.0]
cizgi_kilidi = threading.Lock()


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

# Aynı anda tek bir kare işlensin diye.
isleme_kilidi = threading.Lock()

# ------------------------------------------------------------
# CANLI YAYIN DEPOSU
# ------------------------------------------------------------
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
# ÇİZGİ GEOMETRİSİ (iki nokta -> yön, normal, uzunluk)
# ============================================================

def cizgi_parametreleri():
    """Şu anki çizgiden (p1,p2) yön vektörü, normal vektör ve uzunluk üretir.
    Çizgi dejenere (iki nokta aynı) ise varsayılan yatay çizgiye döner."""
    with cizgi_kilidi:
        x1, y1 = cizgi_p1
        x2, y2 = cizgi_p2

    dx, dy = x2 - x1, y2 - y1
    uzunluk = (dx ** 2 + dy ** 2) ** 0.5

    if uzunluk < 1e-6:
        x1, y1 = 0.0, FRAME_H * 0.6
        x2, y2 = float(FRAME_W), FRAME_H * 0.6
        dx, dy = x2 - x1, y2 - y1
        uzunluk = (dx ** 2 + dy ** 2) ** 0.5

    ux, uy = dx / uzunluk, dy / uzunluk
    nx, ny = -uy, ux  # normal (çizgiye dik) birim vektör

    return x1, y1, x2, y2, ux, uy, nx, ny, uzunluk


def nokta_cizgi_uzerinde_mi(px, py, params):
    """Nokta, çizginin iki ucundaki 'ölü alan' payı hariç, çizginin
    kapsadığı aralık içinde mi? (Ölü alan çizgiyle birlikte hareket eder.)"""
    x1, y1, x2, y2, ux, uy, nx, ny, uzunluk = params
    t = ((px - x1) * ux + (py - y1) * uy) / uzunluk
    kenar_pay_t = KENAR_PAY / uzunluk
    return kenar_pay_t <= t <= (1 - kenar_pay_t)


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
    global KARE_SAYAC

    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    annotated_frame = frame.copy()

    with isleme_kilidi:
       

        params = cizgi_parametreleri()
        x1, y1, x2, y2, ux, uy, nx, ny, uzunluk = params

        # Ana çizgi
        cv2.line(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
        # Histerezis paralel çizgileri (ana çizgiye dik yönde kaydırılmış)
        p1a = (int(x1 + nx * HISTEREZIS_PAY), int(y1 + ny * HISTEREZIS_PAY))
        p2a = (int(x2 + nx * HISTEREZIS_PAY), int(y2 + ny * HISTEREZIS_PAY))
        p1b = (int(x1 - nx * HISTEREZIS_PAY), int(y1 - ny * HISTEREZIS_PAY))
        p2b = (int(x2 - nx * HISTEREZIS_PAY), int(y2 - ny * HISTEREZIS_PAY))
        cv2.line(annotated_frame, p1a, p2a, (0, 255, 255), 1)
        cv2.line(annotated_frame, p1b, p2b, (0, 255, 255), 1)


         KARE_SAYAC += 1
        kare_sayac_simdi = KARE_SAYAC 
        
        try:
            results = model.track(
                frame, classes=[0], persist=True, tracker=BYTETRACK_CFG,
                verbose=False, conf=0.25, iou=0.5
            )
        except Exception:
            return annotated_frame

        result = results[0]
        guncel_yukseklikler = {}

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

                track_son_gorulme[track_id] = time.time()
                guncel_yukseklikler[track_id] = kutu_h

                ayak_x, ayak_y = int((bx1 + bx2) / 2), by2

                cv2.circle(annotated_frame, (ayak_x, ayak_y), 6, (255, 0, 255), -1)
                cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

                # Yaş / Cinsiyet Tahmini
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

                # Boy Hesaplama
                kisi_boyu_m = None
                with kalibrasyon_kilidi:
                    if is_calibrated and pixels_per_meter > 0:
                        kisi_boyu_m = kutu_h / pixels_per_meter

                # Sayım Kontrolü (çizgiye göre, ölü alan çizgiyle birlikte hareket eder)
                cizgi_ici_mi = nokta_cizgi_uzerinde_mi(ayak_x, ayak_y, params)
                guvenilir = (
                    kutu_h >= MIN_SAYIM_KUTU_YUKSEKLIGI
                    and cizgi_ici_mi
                    and 0 <= ayak_y < FRAME_H
                    and 0 <= ayak_x < FRAME_W
                )

                if kare_sayac_simdi % 15 == 0:
                    print(
                        f"[DEBUG] ID={track_id} kutu_h={kutu_h} "
                        f"(esik={MIN_SAYIM_KUTU_YUKSEKLIGI}) "
                        f"cizgi_ici_mi={cizgi_ici_mi} guvenilir={guvenilir} "
                        f"ayak=({ayak_x},{ayak_y})"
                    )

                if guvenilir:
                    anlik_konum = nokta_taraf_bul(ayak_x, ayak_y, params)
                    onceki_durum = durumlar.get(track_id)

                    if onceki_durum is None:
                        if anlik_konum is not None:
                            durumlar[track_id] = anlik_konum
                            print(f"[DEBUG-BASLANGIC] ID={track_id} ilk_taraf={anlik_konum}")
                    elif anlik_konum is not None and anlik_konum != onceki_durum:
                        aday = gecis_adaylari.get(track_id)
                        if aday is not None and aday["yon"] == anlik_konum:
                            aday["sayac"] += 1
                        else:
                            aday = {"yon": anlik_konum, "sayac": 1}
                        gecis_adaylari[track_id] = aday

                        print(f"[DEBUG-GECIS] ID={track_id} onceki={onceki_durum} yeni={anlik_konum} aday_sayac={aday['sayac']}/{GECIS_ONAY_KARESI}")

                        if aday["sayac"] >= GECIS_ONAY_KARESI:
                            yeni_durum = anlik_konum
                            # A -> B : Giris, B -> A : Cikis
                            # (yön ters gelirse admin panelinde iki noktayı
                            # ters sırayla yeniden çizmeniz yeterli)
                            if onceki_durum == "A" and yeni_durum == "B":
                                gecisi_kaydet(track_id, "Giris", kisi_boyu_m)
                            elif onceki_durum == "B" and yeni_durum == "A":
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


@app.route("/cizgi_durumu")
def cizgi_durumu():
    with cizgi_kilidi:
        return jsonify({
            "p1": cizgi_p1,
            "p2": cizgi_p2,
            "genislik": FRAME_W,
            "yukseklik": FRAME_H,
        })


def cizgiyi_kareye_uzat(x1, y1, x2, y2):
    """Verilen iki noktadan geçen çizgiyi, aynı açıyı/konumu koruyarak
    kare (FRAME_W x FRAME_H) sınırlarına kadar uzatır. Böylece admin
    panelinde kısa bir çizgi çizilse bile, gerçek geçiş alanının tamamı
    kapsanmış olur ve ölü alan her zaman kameranın gerçek kenarlarında
    oluşur."""
    dx, dy = x2 - x1, y2 - y1
    uzunluk = (dx ** 2 + dy ** 2) ** 0.5
    if uzunluk < 1e-6:
        return x1, y1, x2, y2

    ux, uy = dx / uzunluk, dy / uzunluk

    t_min, t_max = -1e9, 1e9

    if abs(ux) > 1e-9:
        ta, tb = (0 - x1) / ux, (FRAME_W - x1) / ux
        t_min = max(t_min, min(ta, tb))
        t_max = min(t_max, max(ta, tb))

    if abs(uy) > 1e-9:
        ta, tb = (0 - y1) / uy, (FRAME_H - y1) / uy
        t_min = max(t_min, min(ta, tb))
        t_max = min(t_max, max(ta, tb))

    if t_min > t_max:
        return x1, y1, x2, y2  # kare ile kesişim yok, olduğu gibi bırak

    yeni_x1 = x1 + t_min * ux
    yeni_y1 = y1 + t_min * uy
    yeni_x2 = x1 + t_max * ux
    yeni_y2 = y1 + t_max * uy

    return yeni_x1, yeni_y1, yeni_x2, yeni_y2


@app.route("/cizgi_guncelle", methods=["POST"])
def cizgi_guncelle():
    global cizgi_p1, cizgi_p2
    try:
        veri = request.json
        ham_x1 = float(veri["x1"])
        ham_y1 = float(veri["y1"])
        ham_x2 = float(veri["x2"])
        ham_y2 = float(veri["y2"])

        if ((ham_x2 - ham_x1) ** 2 + (ham_y2 - ham_y1) ** 2) ** 0.5 < 10:
            return jsonify({"durum": "hata", "mesaj": "Çizgi çok kısa"}), 400

        # Çizilen açı/konum korunarak kare sınırlarına kadar uzat
        x1, y1, x2, y2 = cizgiyi_kareye_uzat(ham_x1, ham_y1, ham_x2, ham_y2)

        with cizgi_kilidi:
            cizgi_p1 = [x1, y1]
            cizgi_p2 = [x2, y2]

        # Eski çizgiye göre oluşmuş taraf/geçiş durumlarını sıfırla,
        # yeni çizgiyle tutarsız kalmasınlar.
        durumlar.clear()
        gecis_adaylari.clear()

        print(f"[BILGI] Çizgi güncellendi: ({x1:.0f},{y1:.0f}) -> ({x2:.0f},{y2:.0f})")

        return jsonify({"durum": "ok", "p1": cizgi_p1, "p2": cizgi_p2})
    except Exception as e:
        return jsonify({"durum": "hata", "mesaj": str(e)}), 400


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
            a.admin-link { display:inline-block; margin-top:20px; color:#888; font-size:13px; text-decoration:underline; }
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

        <a class="admin-link" href="/admin">Admin Panel (Giriş Çizgisini Ayarla)</a>

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


@app.route("/admin")
def admin_panel():
    return """
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <title>Admin Panel - Giriş Çizgisi</title>
        <style>
            body { background: #111; color: white; font-family: Arial; text-align: center; margin: 0; padding: 20px; }
            h1 { color: #00ff88; font-size: 22px; }
            .sahne { position: relative; display: inline-block; max-width: 95vw; }
            #referansGoruntu { width: 100%; max-width: 900px; display: block; border: 3px solid #444; border-radius: 8px; background: #000; }
            #cizimKatmani { position: absolute; top: 0; left: 0; width: 100%; height: 100%; cursor: crosshair; }
            .btn { background-color: #00ff88; color: #111; padding: 10px 20px; font-size: 15px; font-weight: bold; border: none; border-radius: 5px; cursor: pointer; margin: 6px; }
            .btn:hover { background-color: #00cc6a; }
            .btn.temizle { background-color: #ff5555; }
            .btn.temizle:hover { background-color: #dd3333; }
            .info { margin-top: 12px; font-size: 14px; color: #ccc; max-width: 700px; margin-left: auto; margin-right: auto; }
            #durumYazisi { margin-top: 10px; font-size: 14px; color: #aaa; }
            a.geri-link { display:inline-block; margin-top:20px; color:#888; font-size:13px; text-decoration:underline; }
        </style>
    </head>
    <body>
        <h1>Giriş Çizgisi Ayarla</h1>
        <p class="info">
            Aşağıdaki görüntü üzerinde iki noktaya tıklayın: önce çizginin başlangıcı, sonra bitişi.
            İki nokta seçildiğinde önizleme çizgisi görünür. "A" tarafından "B" tarafına geçiş
            <b>Giriş</b>, tersi <b>Çıkış</b> sayılır (yön ters gelirse noktaları ters sırayla tekrar çizin).
        </p>

        <div class="sahne">
            <img id="referansGoruntu" alt="Görüntü bekleniyor... (bir cihaz kamerasını kullanmalı)">
            <canvas id="cizimKatmani"></canvas>
        </div>

        <div>
            <button class="btn" onclick="cizgiyiKaydet()">Çizgiyi Kaydet</button>
            <button class="btn temizle" onclick="secimiTemizle()">Seçimi Temizle</button>
        </div>

        <div id="durumYazisi">Mevcut çizgi yükleniyor...</div>

        <a class="geri-link" href="/">← Ana sayfaya dön</a>

        <script>
            const img = document.getElementById('referansGoruntu');
            const canvas = document.getElementById('cizimKatmani');
            const ctx = canvas.getContext('2d');
            const durumYazisi = document.getElementById('durumYazisi');

            let p1 = null, p2 = null;
            let mevcutCizgi = null;

            function katmanBoyutlandir() {
                canvas.width = img.clientWidth;
                canvas.height = img.clientHeight;
                cizimYap();
            }
            window.addEventListener('resize', katmanBoyutlandir);
            img.addEventListener('load', katmanBoyutlandir);

            function ekranKoordUretcek(px, py) {
                // Doğal (frame) koordinatı -> ekran (canvas) koordinatına çevir
                const olcekX = canvas.width / img.naturalWidth;
                const olcekY = canvas.height / img.naturalHeight;
                return [px * olcekX, py * olcekY];
            }

            function cizimYap() {
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                // Sunucudaki mevcut (kayıtlı) çizgiyi soluk çiz
                if (mevcutCizgi) {
                    const [ex1, ey1] = ekranKoordUretcek(mevcutCizgi.p1[0], mevcutCizgi.p1[1]);
                    const [ex2, ey2] = ekranKoordUretcek(mevcutCizgi.p2[0], mevcutCizgi.p2[1]);
                    ctx.strokeStyle = 'rgba(255,0,0,0.4)';
                    ctx.lineWidth = 2;
                    ctx.setLineDash([6, 4]);
                    ctx.beginPath();
                    ctx.moveTo(ex1, ey1);
                    ctx.lineTo(ex2, ey2);
                    ctx.stroke();
                    ctx.setLineDash([]);
                }

                // Kullanıcının şu an seçtiği yeni noktalar/çizgi
                if (p1) {
                    ctx.fillStyle = '#00ff88';
                    ctx.beginPath();
                    ctx.arc(p1.ekranX, p1.ekranY, 6, 0, Math.PI * 2);
                    ctx.fill();
                }
                if (p2) {
                    ctx.fillStyle = '#3399ff';
                    ctx.beginPath();
                    ctx.arc(p2.ekranX, p2.ekranY, 6, 0, Math.PI * 2);
                    ctx.fill();
                }
                if (p1 && p2) {
                    ctx.strokeStyle = '#ffff00';
                    ctx.lineWidth = 3;
                    ctx.beginPath();
                    ctx.moveTo(p1.ekranX, p1.ekranY);
                    ctx.lineTo(p2.ekranX, p2.ekranY);
                    ctx.stroke();
                }
            }

            canvas.addEventListener('click', (e) => {
                const rect = canvas.getBoundingClientRect();
                const ekranX = e.clientX - rect.left;
                const ekranY = e.clientY - rect.top;

                const olcekX = img.naturalWidth / canvas.width;
                const olcekY = img.naturalHeight / canvas.height;
                const dogalX = ekranX * olcekX;
                const dogalY = ekranY * olcekY;

                if (!p1) {
                    p1 = { ekranX, ekranY, x: dogalX, y: dogalY };
                } else if (!p2) {
                    p2 = { ekranX, ekranY, x: dogalX, y: dogalY };
                } else {
                    p1 = { ekranX, ekranY, x: dogalX, y: dogalY };
                    p2 = null;
                }
                cizimYap();
            });

            function secimiTemizle() {
                p1 = null;
                p2 = null;
                cizimYap();
            }

            async function cizgiyiKaydet() {
                if (!p1 || !p2) {
                    alert("Önce iki nokta seçmelisiniz.");
                    return;
                }
                try {
                    const r = await fetch('/cizgi_guncelle', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y })
                    });
                    const veri = await r.json();
                    if (veri.durum === 'ok') {
                        durumYazisi.textContent = "Çizgi kaydedildi.";
                        durumYazisi.style.color = "#00ff88";
                        mevcutCizgi = { p1: veri.p1, p2: veri.p2 };
                        p1 = null;
                        p2 = null;
                        cizimYap();
                    } else {
                        alert("Hata: " + (veri.mesaj || "bilinmeyen hata"));
                    }
                } catch (err) {
                    alert("Sunucuya ulaşılamadı: " + err);
                }
            }

            async function referansGoruntuyuGuncelle() {
                try {
                    const r = await fetch('/son_kare');
                    const veri = await r.json();
                    if (veri.image) {
                        img.src = veri.image;
                    }
                } catch (e) {}
            }

            async function mevcutCizgiyiYukle() {
                try {
                    const r = await fetch('/cizgi_durumu');
                    const veri = await r.json();
                    mevcutCizgi = { p1: veri.p1, p2: veri.p2 };
                    durumYazisi.textContent = "Mevcut çizgi: (" + veri.p1[0].toFixed(0) + "," + veri.p1[1].toFixed(0) + ") - (" + veri.p2[0].toFixed(0) + "," + veri.p2[1].toFixed(0) + ")";
                    durumYazisi.style.color = "#aaa";
                    cizimYap();
                } catch (e) {}
            }

            referansGoruntuyuGuncelle();
            setInterval(referansGoruntuyuGuncelle, 1000);
            mevcutCizgiyiYukle();
            setInterval(mevcutCizgiyiYukle, 5000);
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
