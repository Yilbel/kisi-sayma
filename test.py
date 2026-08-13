import os
import time
import threading
from datetime import datetime, timedelta

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, Response
from ultralytics import YOLO
from PIL import Image
from transformers import pipeline


# ============================================================
# GENEL AYARLAR
# ============================================================

PROGRAM_ADI = "Kisi Sayma ve Boy Olcum Sistemi"

FRAME_W = 640
FRAME_H = 480

DROIDCAM_IP = "172.20.10.2"
DROIDCAM_PORT = "4747"

DROIDCAM_URL = (
    f"http://{DROIDCAM_IP}:{DROIDCAM_PORT}/video"
)

WEBCAM_INDEX = 0

FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000


# ============================================================
# KAPI / ÇİZGİ SAYIM AYARLARI
# ============================================================

CIZGI_Y = 420
HISTEREZIS_PAY = 20
GECIS_ONAY_KARESI = 4
MIN_SAYIM_KUTU_YUKSEKLIGI = 60
KENAR_PAY = 10
TRACK_KAYIP_SURESI = 45


# ============================================================
# BOY KALİBRASYONU
# ============================================================

# Kamera karşısındaki referans kişinin gerçek boyu (metre).
KNOWN_USER_HEIGHT_M = 1.75

pixels_per_meter = 100.0
is_calibrated = False

kalibrasyon_kilidi = threading.Lock()


# ============================================================
# RAPOR
# ============================================================

RAPOR_KLASORU = "raporlar"


# ============================================================
# BYTE TRACK AYARI
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

BYTETRACK_CFG = os.path.join(
    BASE_DIR,
    "bytetrack_custom.yaml"
)


# ============================================================
# GLOBAL DURUMLAR
# ============================================================

program_calisiyor = True

giris_sayisi = 0
cikis_sayisi = 0

KARE_SAYAC = 0

durumlar = {}
gecis_adaylari = {}
kisi_bilgisi = {}
deneme_sayisi = {}

MAX_DENEME = 10
TAHMIN_HER_N_KAREDE_BIR = 5
MIN_KUTU_BOYUTU = 60

gecis_kayitlari = []
gecis_kilidi = threading.Lock()

son_frame = None
frame_kilidi = threading.Lock()

son_kutu_yukseklikleri = {}
son_kutu_kilidi = threading.Lock()

track_son_gorulme = {}


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


if not os.path.isfile(BYTETRACK_CFG):
    BYTETRACK_CFG = "bytetrack.yaml"


# ============================================================
# KAMERA
# ============================================================

cap = cv2.VideoCapture(DROIDCAM_URL)
if not cap.isOpened():
    cap = cv2.VideoCapture(WEBCAM_INDEX)

if not cap.isOpened():
    raise RuntimeError("Hiçbir kamera açılamadı!")

print("[OK] Kamera bağlantısı başarılı.")


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
# KALİBRASYON FONKSİYONU
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
    global program_calisiyor
    while program_calisiyor:
        try:
            girdi = input().strip().lower()
            if girdi == "c":
                kalibrasyonu_uygula()
            elif girdi == "q":
                program_calisiyor = False
                break
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
        tid
        for tid, zmn in track_son_gorulme.items()
        if simdi - zmn > TRACK_KAYIP_SURESI
    ]
    for tid in silinecekler:
        track_son_gorulme.pop(tid, None)
        durumlar.pop(tid, None)
        gecis_adaylari.pop(tid, None)
        kisi_bilgisi.pop(tid, None)
        deneme_sayisi.pop(tid, None)


# ============================================================
# KAMERA DÖNGÜSÜ
# ============================================================

def kamera_dongusu():
    global program_calisiyor, KARE_SAYAC, son_frame

    while program_calisiyor:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        KARE_SAYAC += 1

        try:
            results = model.track(
                frame,
                classes=[0],
                persist=True,
                tracker=BYTETRACK_CFG,
                verbose=False,
                conf=0.35,
                iou=0.5
            )
        except Exception:
            continue

        result = results[0]
        annotated_frame = frame.copy()

        # Çizgileri çiz
        cv2.line(annotated_frame, (0, CIZGI_Y), (FRAME_W, CIZGI_Y), (0, 0, 255), 3)
        cv2.line(annotated_frame, (0, CIZGI_Y - HISTEREZIS_PAY), (FRAME_W, CIZGI_Y - HISTEREZIS_PAY), (0, 255, 255), 1)
        cv2.line(annotated_frame, (0, CIZGI_Y + HISTEREZIS_PAY), (FRAME_W, CIZGI_Y + HISTEREZIS_PAY), (0, 255, 255), 1)

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

                kutu_w = x2 - x1
                kutu_h = y2 - y1
                if kutu_w <= 0 or kutu_h <= 0:
                    continue

                track_son_gorulme[track_id] = time.time()
                guncel_yukseklikler[track_id] = kutu_h

                ayak_x = int((x1 + x2) / 2)
                ayak_y = y2

                cv2.circle(annotated_frame, (ayak_x, ayak_y), 6, (255, 0, 255), -1)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Yaş / Cinsiyet Tahmini
                if track_id not in kisi_bilgisi:
                    deneme_sayisi.setdefault(track_id, 0)
                    if (
                        deneme_sayisi[track_id] < MAX_DENEME
                        and KARE_SAYAC % TAHMIN_HER_N_KAREDE_BIR == 0
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

                # Ekran Etiketi
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

        if KARE_SAYAC % 30 == 0:
            eski_trackleri_temizle()

        # Üst Bilgi Paneli
        cv2.rectangle(annotated_frame, (0, 0), (640, 105), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"GIRIS : {giris_sayisi}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"CIKIS : {cikis_sayisi}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)

        with kalibrasyon_kilidi:
            px_text = f"{pixels_per_meter:.0f}px/m" if is_calibrated else "--px/m"
        cv2.putText(annotated_frame, px_text, (20, FRAME_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        with frame_kilidi:
            son_frame = annotated_frame.copy()

        try:
            cv2.imshow("Kisi Sayma ve Boy Olcum", annotated_frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                program_calisiyor = False
                break
        except cv2.error:
            pass

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# FLASK & RAPOR (Aynı Yapı)
# ============================================================

app = Flask(__name__)

def frame_uret():
    while program_calisiyor:
        with frame_kilidi:
            if son_frame is None:
                time.sleep(0.01)
                continue
            frame = son_frame.copy()
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

@app.route("/video")
def video():
    return Response(frame_uret(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/")
def anasayfa():
    return """
    <!DOCTYPE html><html lang="tr"><head><meta charset="UTF-8"><title>Kişi Sayma ve Boy Ölçüm</title>
    <style>body{background:#111;color:white;font-family:Arial;text-align:center;margin:0;padding:20px;}h1{color:#00ff88;}img{width:640px;max-width:95vw;border:3px solid #444;}</style></head>
    <body><h1>Kişi Sayma ve Boy Ölçüm Sistemi</h1><img src="/video">
    <p>Kalibrasyon için terminale <b>c + Enter</b> yazın.</p></body></html>
    """

if __name__ == "__main__":
    threading.Thread(target=kamera_dongusu, daemon=True).start()
    threading.Thread(target=klavye_dinleyici, daemon=True).start()

    try:
        app.run(host=FLASK_HOST, port=FLASK_PORT, threaded=True, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        program_calisiyor = False