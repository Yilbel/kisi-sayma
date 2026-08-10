import faulthandler
import os
import sys
import time
import cv2
import csv
import traceback
import numpy as np
from datetime import datetime
from flask import Flask, Response

os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

from ultralytics import YOLO
import threading

DeepFace = None

faulthandler.enable(all_threads=True)

# ==========================================================
# TABLET VE DROIDCAM AYARLARI
# ==========================================================
TABLET_IP = "192.168.1.104"
PORT = "4747"
DROIDCAM_URL = f"http://{TABLET_IP}:{PORT}/video"
# ==========================================================

print(f"\n---> Tablet kamerasina baglaniliyor: {DROIDCAM_URL}")
print("---> Lutfen bekleyin...\n")

MODEL_PATH = "yolov8n.pt"
if not os.path.isfile(MODEL_PATH):
    print(f"[BILGI] {MODEL_PATH} bulunmuyor. Uzak model indiriliyor...")
    model = YOLO("yolov8n")
else:
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"[BILGI] {MODEL_PATH} yuklenemedi: {e}. Uzak model indiriliyor...")
        model = YOLO("yolov8n")


def warmup_yolo_model():
    try:
        print("[BILGI] YOLO model hazirlik islemleri yapiliyor...")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = model.predict(dummy_frame, classes=[0], verbose=False)
        print("[BILGI] YOLO model hazirligi tamamlandi.")
    except Exception as e:
        print(f"[UYARI] YOLO model hazirligi sirasinda hata: {e}")
        traceback.print_exc()


def open_droidcam_stream(url, attempts=5, delay=2):
    backends = [None]
    if hasattr(cv2, 'CAP_FFMPEG'):
        backends.append(cv2.CAP_FFMPEG)
    for attempt in range(1, attempts + 1):
        for backend in backends:
            backend_name = 'default' if backend is None else 'FFMPEG'
            print(f"[BILGI] DroidCam stream baglanti denemesi {attempt}/{attempts} ({backend_name}): {url}")
            cap = cv2.VideoCapture(url) if backend is None else cv2.VideoCapture(url, backend)
            if cap.isOpened():
                print(f"[BILGI] DroidCam stream baglanti basarili ({backend_name}).")
                return cap
            cap.release()
            print(f"[UYARI] {backend_name} backend ile DroidCam stream acilamadi.")
        print(f"[UYARI] Tüm backendler başarısız oldu, {delay} saniye sonra tekrar denenecek.")
        time.sleep(delay)
    return None


def reopen_droidcam_stream():
    global cap
    print("[BILGI] DroidCam baglantisi yeniden kuruluyor...")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    cap = open_droidcam_stream(DROIDCAM_URL)
    if cap is not None and cap.isOpened():
        return True
    print("[UYARI] DroidCam baglantisi yeniden kurulamadı.")
    return False

cap = None

if not reopen_droidcam_stream():
    print("[HATA] Tablete baglanilamadi! Bilgisayar kamerasina geciliyor...")
    cap = cv2.VideoCapture(0)

giris_sayisi = 0
cikis_sayisi = 0
durumlar = {}

# ---- YAS TESPITI ICIN EKLENEN KISIM ----
yas_gruplari = {}
deneme_sayisi = {}
MAX_DENEME = 10
KARE_SAYAC = 0
DEEPFACE_HER_N_KAREDE_BIR = 2
MIN_KUTU_BOYUTU = 60

gecis_kayitlari = []

son_frame = None
raw_frame = None
frame_kilidi = threading.Lock()
raw_frame_kilidi = threading.Lock()


def yas_grubuna_cevir(yas):
    if yas <= 12:
        return "Cocuk"
    elif yas <= 24:
        return "Genc"
    elif yas <= 59:
        return "Yetiskin"
    else:
        return "Yasli"


def yas_tahmin_et(frame, box, track_id=None):
    global DeepFace
    if DeepFace is None:
        try:
            from deepface import DeepFace as _DeepFace
            DeepFace = _DeepFace
        except Exception as e:
            print(f"[HATA] DeepFace import sirasinda hata: {e}")
            traceback.print_exc()
            return None

    x1, y1, x2, y2 = [max(0, int(v)) for v in box]
    kirpilmis = frame[y1:y2, x1:x2]

    if kirpilmis.size == 0:
        return None

    debug_dosya = f"debug_kirpilmis_ID{track_id}_deneme{deneme_sayisi.get(track_id, 0)}.jpg"
    cv2.imwrite(debug_dosya, kirpilmis)

    try:
        sonuc = DeepFace.analyze(
            kirpilmis,
            actions=["age"],
            detector_backend="opencv",
            enforce_detection=False,
            silent=True,
        )
        if isinstance(sonuc, list):
            sonuc = sonuc[0]
        yas = sonuc["age"]
        return yas_grubuna_cevir(yas)
    except Exception as e:
        print(f"[YAS TAHMIN HATASI - ID {track_id} icin]: {e}")
        traceback.print_exc()
        return None


CIZGI_X = 320


def capture_thread():
    global cap, raw_frame
    print("[BILGI] capture_thread basladi.")
    while True:
        try:
            if cap is None or not getattr(cap, 'isOpened', lambda: False)():
                if not reopen_droidcam_stream():
                    time.sleep(2)
                    continue

            ret, frame = cap.read()
        except Exception as e:
            print(f"[HATA] capture_thread cap.read() sirasinda istisna: {e}")
            traceback.print_exc()
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
            time.sleep(2)
            continue

        if not ret or frame is None:
            print("[UYARI] Kare okunamadi veya frame None. DroidCam baglantisi yeniden deneniyor...")
            if reopen_droidcam_stream():
                continue
            time.sleep(2)
            continue

        try:
            with raw_frame_kilidi:
                raw_frame = frame.copy()
        except Exception as e:
            print(f"[HATA] capture_thread frame.copy() sirasinda istisna: {e}")
            traceback.print_exc()
            time.sleep(0.1)
            continue

        time.sleep(0.01)


def kamera_dongusu():
    global giris_sayisi, cikis_sayisi, KARE_SAYAC, son_frame
    print("[BILGI] kamera_dongusu thread basladi.")

    while True:
        try:
            with raw_frame_kilidi:
                frame = None if raw_frame is None else raw_frame.copy()

            if frame is None:
                time.sleep(0.1)
                continue

            try:
                frame = cv2.resize(frame, (640, 480))
            except Exception as e:
                print(f"[HATA] Kare boyutlandirma sirasinda hata: {e}. Yeniden denenecek.")
                traceback.print_exc()
                time.sleep(1)
                continue

            if frame is None:
                time.sleep(0.1)
                continue

            with frame_kilidi:
                son_frame = frame.copy()

            KARE_SAYAC += 1
        except Exception as e:
            print(f"[HATA] kamera_dongusu ana dongu istisnasi: {e}")
            traceback.print_exc()
            time.sleep(1)
            continue

        try:
            print(f"[BILGI] YOLO tahminine hazirlaniliyor. KARE_SAYAC={KARE_SAYAC}")
            results = model.track(frame, classes=[0], persist=True, verbose=False)
            annotated_frame = results[0].plot(labels=False, conf=False)
            print("[BILGI] YOLO tahmini basarili.")
        except Exception as e:
            print(f"[HATA] YOLO tahmini sirasinda istisna: {e}")
            traceback.print_exc()
            continue

        try:
            cv2.line(annotated_frame, (CIZGI_X, 0), (CIZGI_X, annotated_frame.shape[0]), (0, 0, 255), 2)
        except Exception as e:
            print(f"[HATA] annotate islemi sirasinda istisna: {e}")
            traceback.print_exc()
            continue

        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.int().tolist()
            boxes = results[0].boxes.xyxy.tolist()

            for track_id, box in zip(ids, boxes):
                x1, y1, x2, y2 = [int(v) for v in box]
                merkez_x = int((x1 + x2) / 2)

                if track_id not in yas_gruplari:
                    deneme_sayisi.setdefault(track_id, 0)

                    kutu_yeterince_buyuk = (x2 - x1) >= MIN_KUTU_BOYUTU and (y2 - y1) >= MIN_KUTU_BOYUTU

                    kareyi_dene_mi = (
                        deneme_sayisi[track_id] < MAX_DENEME
                        and KARE_SAYAC % DEEPFACE_HER_N_KAREDE_BIR == 0
                        and kutu_yeterince_buyuk
                    )

                    if kareyi_dene_mi:
                        tahmin = yas_tahmin_et(frame, (x1, y1, x2, y2), track_id)
                        deneme_sayisi[track_id] += 1

                        if tahmin is not None:
                            yas_gruplari[track_id] = tahmin
                        elif deneme_sayisi[track_id] >= MAX_DENEME:
                            yas_gruplari[track_id] = "Bilinmiyor"

                if merkez_x < CIZGI_X:
                    yeni_durum = "sol"
                else:
                    yeni_durum = "sag"

                onceki_durum = durumlar.get(track_id)

                if onceki_durum is not None and onceki_durum != yeni_durum:
                    kisinin_yas_grubu = yas_gruplari.get(track_id, "Bilinmiyor")

                    if onceki_durum == "sol" and yeni_durum == "sag":
                        giris_sayisi += 1
                        gecis_kayitlari.append({
                            "zaman": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "tip": "Giris",
                            "yas_grubu": kisinin_yas_grubu,
                        })
                    elif onceki_durum == "sag" and yeni_durum == "sol":
                        cikis_sayisi += 1
                        gecis_kayitlari.append({
                            "zaman": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "tip": "Cikis",
                            "yas_grubu": kisinin_yas_grubu,
                        })

                durumlar[track_id] = yeni_durum

                etiket = f"ID: {track_id}"
                if track_id in yas_gruplari:
                    etiket += f" | {yas_gruplari[track_id]}"
                cv2.putText(annotated_frame, etiket, (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(annotated_frame, f"Giris: {giris_sayisi}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Cikis: {cikis_sayisi}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        with frame_kilidi:
            son_frame = annotated_frame.copy()


# ==========================================================
# FLASK WEB ARAYUZU
# ==========================================================
app = Flask(__name__)


def frame_uret():
    while True:
        with frame_kilidi:
            if son_frame is None:
                frame_bytes = None
            else:
                basarili, buffer = cv2.imencode('.jpg', son_frame)
                if basarili:
                    frame_bytes = buffer.tobytes()
                else:
                    frame_bytes = None

        if frame_bytes is None:
            time.sleep(0.05)
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video')
def video():
    return Response(frame_uret(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def anasayfa():
    return f"""
    <html>
    <head><title>Kisi Sayma</title></head>
    <body style="text-align:center; font-family:sans-serif;">
        <h1>Kisi Sayma Sistemi</h1>
        <p>Giris: {giris_sayisi} | Cikis: {cikis_sayisi}</p>
        <img src="/video" width="640" height="480">
    </body>
    </html>
    """


if __name__ == '__main__':
    warmup_yolo_model()

    capture_thread_inst = threading.Thread(target=capture_thread, daemon=True)
    capture_thread_inst.start()

    kamera_thread = threading.Thread(target=kamera_dongusu, daemon=True)
    kamera_thread.start()

    app.run(host='0.0.0.0', port=5000)
