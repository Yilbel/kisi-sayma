import os
import time
import cv2
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, Response
from ultralytics import YOLO
from PIL import Image
from transformers import pipeline
import threading
import matplotlib
matplotlib.use("Agg")  # ekransiz grafik uretimi icin
import matplotlib.pyplot as plt

# ==========================================================
# TABLET VE DROIDCAM AYARLARI
# ==========================================================
TABLET_IP = "192.168.1.104"
PORT = "4747"
DROIDCAM_URL = f"http://{TABLET_IP}:{PORT}/video"
# ==========================================================

# ==========================================================
# BOY (YUKSEKLIK) KALIBRASYON AYARLARI
# ----------------------------------------------------------
# Farkli bir referans boyla kalibrasyon yapmak isterseniz
# (orn. 1.55m) SADECE asagidaki satiri degistirmeniz yeterli.
# 'c' tusuna (veya terminalde 'c' + Enter) basildiginda, o an
# kamerada gorunen kisinin piksel boyu bu deger uzerinden
# pixels_per_meter'e cevrilip kilitlenir.
# ==========================================================
KNOWN_USER_HEIGHT_M = 1.75  # <-- kalibrasyon referans boyu (metre) - TEK DEGISECEK YER
pixels_per_meter = 100.0
is_calibrated = False
kalibrasyon_kilidi = threading.Lock()
# ==========================================================

# ==========================================================
# ORTA CIZGI - YATAY (sadece asagi <-> yukari gecisine gore sayim)
# ==========================================================
CIZGI_Y = 240
FRAME_W, FRAME_H = 640, 480
KENAR_PAY = 15              # kenara bu kadar piksel kala tespit guvenilmez sayilir
MIN_SAYIM_KUTU_YUKSEKLIK = 80  # sayim icin kutunun en az bu yukseklikte olmasi gerekir
# ==========================================================

# ==========================================================
# RAPOR AYARLARI
# ==========================================================
RAPOR_KLASORU = "raporlar"
# ==========================================================

print(f"\n---> Tablet kamerasina baglaniliyor: {DROIDCAM_URL}")
print("---> Lutfen bekleyin...\n")

# ---- YUZ TESPITI (OpenCV Haar Cascade) ----
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---- YAS VE CINSIYET MODELLERI (Apache 2.0 - ticari kullanima uygun) ----
print("[BILGI] Yas/cinsiyet modelleri yukleniyor (ilk calistirmada indirilecek)...")
age_pipe = pipeline("image-classification", model="dima806/fairface_age_image_detection")
gender_pipe = pipeline("image-classification", model="dima806/fairface_gender_image_detection")
print("[BILGI] Modeller yuklendi.\n")


def yas_bandini_gruba_cevir(bant: str) -> str:
    cocuk = {"0-2", "3-9"}
    genc = {"10-19", "20-29"}
    yetiskin = {"30-39", "40-49", "50-59", "60-69", "70+"}
    if bant in cocuk:
        return "Cocuk"
    elif bant in genc:
        return "Genc"
    elif bant in yetiskin:
        return "Yetiskin"
    return "Bilinmiyor"


def cinsiyet_cevir(label: str) -> str:
    label = label.lower()
    if "female" in label or "kadin" in label:
        return "Kadin"
    if "male" in label or "erkek" in label:
        return "Erkek"
    return "Bilinmiyor"


# ---- YOLO MODELI (kisi tespiti/takibi) ----
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

# DroidCam baglantisi
cap = cv2.VideoCapture(DROIDCAM_URL)
if not cap.isOpened():
    print("[HATA] Tablete baglanilamadi! Bilgisayar kamerasina geciliyor...")
    cap = cv2.VideoCapture(0)

giris_sayisi = 0
cikis_sayisi = 0
durumlar = {}  # track_id -> "yukari" | "asagi"  (sadece guvenilir kutularla guncellenir)

# ---- YAS/CINSIYET TESPIT AYARLARI ----
kisi_bilgisi = {}       # track_id -> {"yas_grubu": ..., "cinsiyet": ...}
deneme_sayisi = {}
MAX_DENEME = 10
KARE_SAYAC = 0
TAHMIN_HER_N_KAREDE_BIR = 3
MIN_KUTU_BOYUTU = 60
MIN_YUZ_BOYUTU = 40   # bulunan yuz kirpmasi bu boyuttan kucukse guvenilmez sayilir

gecis_kayitlari = []
gecis_kilidi = threading.Lock()

son_frame = None
frame_kilidi = threading.Lock()

# En son gorulen kutu yukseklikleri ('c' ile kalibrasyon bunu kullanir): track_id -> piksel yukseklik
son_kutu_yukseklikleri = {}
son_kutu_kilidi = threading.Lock()


def yuz_bul(kirpilmis_govde):
    """Govde/kafa kirpmasi icinde Haar Cascade ile gercek yuzu bulur."""
    if kirpilmis_govde.size == 0:
        return None
    gri = cv2.cvtColor(kirpilmis_govde, cv2.COLOR_BGR2GRAY)
    yuzler = face_cascade.detectMultiScale(gri, scaleFactor=1.1, minNeighbors=5,
                                            minSize=(MIN_YUZ_BOYUTU, MIN_YUZ_BOYUTU))
    if len(yuzler) == 0:
        return None
    fx, fy, fw, fh = max(yuzler, key=lambda r: r[2] * r[3])
    return kirpilmis_govde[fy:fy + fh, fx:fx + fw]


def yas_cinsiyet_tahmin_et(frame, box, track_id=None):
    x1, y1, x2, y2 = [max(0, int(v)) for v in box]
    yukseklik = y2 - y1
    ust_y2 = y1 + int(yukseklik * 0.55)
    kirpilmis_govde = frame[y1:ust_y2, x1:x2]

    yuz = yuz_bul(kirpilmis_govde)
    if yuz is None or yuz.size == 0:
        return None, None

    try:
        rgb_yuz = cv2.cvtColor(yuz, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_yuz)
        age_sonuc = age_pipe(pil_img)[0]
        gender_sonuc = gender_pipe(pil_img)[0]
        yas_grubu = yas_bandini_gruba_cevir(age_sonuc["label"])
        cinsiyet = cinsiyet_cevir(gender_sonuc["label"])
        print(f"[LOG] ID {track_id} -> Yas bandi: {age_sonuc['label']} "
              f"({yas_grubu}) | Cinsiyet: {cinsiyet}")
        return yas_grubu, cinsiyet
    except Exception as e:
        print(f"[TAHMIN HATASI - ID {track_id}]: {e}")
        return None, None


def kalibrasyonu_uygula():
    """
    'c' tusuna (pencere odaktayken) veya terminalde 'c'+Enter yazildiginda cagrilir.
    Su an kamerada gorunen en yuksek (piksel olarak en buyuk) kutuyu referans alip,
    KNOWN_USER_HEIGHT_M degerine gore pixels_per_meter'i kilitler.
    """
    global pixels_per_meter, is_calibrated
    with son_kutu_kilidi:
        if not son_kutu_yukseklikleri:
            print("[UYARI] Kalibrasyon icin kameranin onunde gorunur bir kisi olmali!")
            return
        track_id, yukseklik_px = max(son_kutu_yukseklikleri.items(), key=lambda kv: kv[1])

    with kalibrasyon_kilidi:
        pixels_per_meter = yukseklik_px / KNOWN_USER_HEIGHT_M
        is_calibrated = True

    print(f"[BASARILI] Kalibrasyon tamamlandi! ID {track_id} referans alindi "
          f"(referans boy: {KNOWN_USER_HEIGHT_M} m), "
          f"1 metre = {pixels_per_meter:.2f} piksel olarak kilitlendi.")


def klavye_dinleyici():
    """
    Terminalden 'c' yazip Enter'a basarak da kalibrasyon tetiklenebilir.
    Onizleme penceresi odakta olmasa/gorunmese bile kalibrasyonun
    calismasini garanti eder.
    """
    print("[BILGI] Kalibrasyon icin: onizleme penceresi acikken 'c' tusuna basin, "
          "YA DA bu terminale 'c' yazip Enter'a basin.")
    while True:
        try:
            girdi = input()
        except EOFError:
            break
        if girdi.strip().lower() == 'c':
            kalibrasyonu_uygula()


# ==========================================================
# AYLIK RAPOR OLUSTURMA (sadece dosyaya - ekrana YAZILMAZ)
# ==========================================================
def aylik_rapor_olustur(yil=None, ay=None):
    """
    Verilen yil/ay icin (varsayilan: bu ay) aylik rapor dosyalarini olusturur:
      - aylik_rapor_YYYY_MM.txt  (saatlik yas/cinsiyet dagilimi, yuzdeler, en yogun saatler)
      - aylik_rapor_YYYY_MM.png  (ayni verinin grafikleri)
    Sadece 'Giris' kayitlarini temel alir.
    """
    simdi = datetime.now()
    hedef_yil = yil or simdi.year
    hedef_ay = ay or simdi.month

    with gecis_kilidi:
        kayit_kopyasi = list(gecis_kayitlari)

    giris_kayitlari = []
    for k in kayit_kopyasi:
        zaman = datetime.strptime(k["zaman"], "%Y-%m-%d %H:%M:%S")
        if k["tip"] == "Giris" and zaman.year == hedef_yil and zaman.month == hedef_ay:
            giris_kayitlari.append((zaman, k))

    if not giris_kayitlari:
        print(f"[RAPOR] {hedef_yil}-{hedef_ay:02d} icin kayit yok, rapor olusturulmadi.")
        return

    toplam = len(giris_kayitlari)
    saat_yas = {}
    saat_cinsiyet = {}
    cinsiyet_toplam = {}
    saat_toplam = {}

    for zaman, k in giris_kayitlari:
        saat = zaman.hour
        yg = k["yas_grubu"]
        cs = k["cinsiyet"]

        saat_yas.setdefault(saat, {})
        saat_yas[saat][yg] = saat_yas[saat].get(yg, 0) + 1

        saat_cinsiyet.setdefault(saat, {})
        saat_cinsiyet[saat][cs] = saat_cinsiyet[saat].get(cs, 0) + 1

        cinsiyet_toplam[cs] = cinsiyet_toplam.get(cs, 0) + 1
        saat_toplam[saat] = saat_toplam.get(saat, 0) + 1

    # ---- METIN RAPORU ----
    satirlar = []
    satirlar.append(f"===== AYLIK RAPOR: {hedef_yil}-{hedef_ay:02d} =====")
    satirlar.append(f"Toplam Giris Sayisi: {toplam}\n")

    satirlar.append("---- SAATLERE GORE YAS GRUBU DAGILIMI ----")
    for saat in sorted(saat_yas.keys()):
        detay = ", ".join(f"{yg}: {sayi}" for yg, sayi in sorted(saat_yas[saat].items()))
        satirlar.append(f"{saat:02d}:00 -> {detay}")

    satirlar.append("\n---- SAATLERE GORE CINSIYET DAGILIMI ----")
    for saat in sorted(saat_cinsiyet.keys()):
        detay = ", ".join(f"{cs}: {sayi}" for cs, sayi in sorted(saat_cinsiyet[saat].items()))
        satirlar.append(f"{saat:02d}:00 -> {detay}")

    satirlar.append("\n---- AYLIK CINSIYET YUZDELERI ----")
    for cs, sayi in sorted(cinsiyet_toplam.items(), key=lambda kv: kv[1], reverse=True):
        yuzde = (sayi / toplam) * 100
        satirlar.append(f"{cs}: %{yuzde:.1f} ({sayi} kisi)")

    satirlar.append("\n---- EN YOGUN SAATLER (YUZDE) ----")
    for saat, sayi in sorted(saat_toplam.items(), key=lambda kv: kv[1], reverse=True):
        yuzde = (sayi / toplam) * 100
        satirlar.append(f"{saat:02d}:00 -> %{yuzde:.1f} ({sayi} kisi)")

    os.makedirs(RAPOR_KLASORU, exist_ok=True)
    txt_yolu = os.path.join(RAPOR_KLASORU, f"aylik_rapor_{hedef_yil}_{hedef_ay:02d}.txt")
    with open(txt_yolu, "w", encoding="utf-8") as f:
        f.write("\n".join(satirlar))

    # ---- GRAFIKLER ----
    try:
        tum_saatler = list(range(24))
        tum_yas_gruplari = sorted({yg for d in saat_yas.values() for yg in d.keys()})
        tum_cinsiyetler = sorted({cs for d in saat_cinsiyet.values() for cs in d.keys()})

        fig, eksenler = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Aylik Rapor: {hedef_yil}-{hedef_ay:02d}  (Toplam Giris: {toplam})", fontsize=14)

        ax = eksenler[0][0]
        alt_taban = np.zeros(len(tum_saatler))
        for yg in tum_yas_gruplari:
            degerler = np.array([saat_yas.get(s, {}).get(yg, 0) for s in tum_saatler])
            ax.bar(tum_saatler, degerler, bottom=alt_taban, label=yg)
            alt_taban += degerler
        ax.set_title("Saatlere Gore Yas Grubu")
        ax.set_xlabel("Saat")
        ax.set_ylabel("Kisi Sayisi")
        ax.legend()

        ax = eksenler[0][1]
        alt_taban = np.zeros(len(tum_saatler))
        for cs in tum_cinsiyetler:
            degerler = np.array([saat_cinsiyet.get(s, {}).get(cs, 0) for s in tum_saatler])
            ax.bar(tum_saatler, degerler, bottom=alt_taban, label=cs)
            alt_taban += degerler
        ax.set_title("Saatlere Gore Cinsiyet")
        ax.set_xlabel("Saat")
        ax.set_ylabel("Kisi Sayisi")
        ax.legend()

        ax = eksenler[1][0]
        etiketler = list(cinsiyet_toplam.keys())
        degerler = list(cinsiyet_toplam.values())
        ax.pie(degerler, labels=etiketler, autopct="%1.1f%%")
        ax.set_title("Aylik Cinsiyet Yuzdesi")

        ax = eksenler[1][1]
        saat_sirali = sorted(saat_toplam.items(), key=lambda kv: kv[0])
        saatler_x = [s for s, _ in saat_sirali]
        yuzdeler_y = [(sayi / toplam) * 100 for _, sayi in saat_sirali]
        ax.bar(saatler_x, yuzdeler_y, color="orange")
        ax.set_title("Saatlik Yogunluk (%)")
        ax.set_xlabel("Saat")
        ax.set_ylabel("Yuzde (%)")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        png_yolu = os.path.join(RAPOR_KLASORU, f"aylik_rapor_{hedef_yil}_{hedef_ay:02d}.png")
        plt.savefig(png_yolu)
        plt.close(fig)
    except Exception as e:
        print(f"[RAPOR] Grafik olusturulamadi: {e}")
        png_yolu = None

    print(f"[RAPOR] Kaydedildi: {txt_yolu}" + (f" ve {png_yolu}" if png_yolu else ""))


def aylik_rapor_zamanlayici():
    """Her saat kontrol eder; ay degisince bir onceki ayin raporunu otomatik dosyaya yazar."""
    son_kontrol_ay = datetime.now().month
    while True:
        time.sleep(3600)
        simdi = datetime.now()
        if simdi.month != son_kontrol_ay:
            onceki_ay_tarih = simdi.replace(day=1) - timedelta(days=1)
            aylik_rapor_olustur(onceki_ay_tarih.year, onceki_ay_tarih.month)
            son_kontrol_ay = simdi.month


def kamera_dongusu():
    global giris_sayisi, cikis_sayisi, KARE_SAYAC, son_frame

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Goruntu alinamadi. Program kapaniyor...")
            break

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        KARE_SAYAC += 1

        results = model.track(frame, classes=[0], persist=True, verbose=False)
        annotated_frame = results[0].plot(labels=False, conf=False)

        # ---- ORTA CIZGI: YATAY ----
        cv2.line(annotated_frame, (0, CIZGI_Y), (annotated_frame.shape[1], CIZGI_Y), (0, 0, 255), 2)

        guncel_yukseklikler = {}

        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.int().tolist()
            boxes = results[0].boxes.xyxy.tolist()

            for track_id, box in zip(ids, boxes):
                x1, y1, x2, y2 = [int(v) for v in box]
                merkez_y = int((y1 + y2) / 2)
                kutu_yuksekligi_px = y2 - y1
                guncel_yukseklikler[track_id] = kutu_yuksekligi_px

                # ---- YAS / CINSIYET TAHMINI ----
                if track_id not in kisi_bilgisi:
                    deneme_sayisi.setdefault(track_id, 0)

                    kutu_yeterince_buyuk = (x2 - x1) >= MIN_KUTU_BOYUTU and (y2 - y1) >= MIN_KUTU_BOYUTU

                    kareyi_dene_mi = (
                        deneme_sayisi[track_id] < MAX_DENEME
                        and KARE_SAYAC % TAHMIN_HER_N_KAREDE_BIR == 0
                        and kutu_yeterince_buyuk
                    )

                    if kareyi_dene_mi:
                        yas_grubu, cinsiyet = yas_cinsiyet_tahmin_et(frame, (x1, y1, x2, y2), track_id)
                        deneme_sayisi[track_id] += 1

                        if yas_grubu is not None and cinsiyet is not None:
                            kisi_bilgisi[track_id] = {"yas_grubu": yas_grubu, "cinsiyet": cinsiyet}
                        elif deneme_sayisi[track_id] >= MAX_DENEME:
                            kisi_bilgisi[track_id] = {"yas_grubu": "Bilinmiyor", "cinsiyet": "Bilinmiyor"}

                # ---- BOY HESABI ----
                kisi_boyu_m = None
                with kalibrasyon_kilidi:
                    if is_calibrated and pixels_per_meter > 0:
                        kisi_boyu_m = kutu_yuksekligi_px / pixels_per_meter

                # ---- GIRIS / CIKIS SAYIMI: SADECE yatay cizgi gecisi ----
                # Kutu ekran kenarina cok yakinsa (kisi sagdan/soldan cikarken kutu
                # kirpiliyor) veya kutu kucukse, bu kare guvenilmez sayilir ve
                # durum/sayim GUNCELLENMEZ. Boylece kenardan cikis-giris hareketi
                # yanlislikla yatay cizgi gecisi gibi sayilmaz.
                guvenilir_kutu = (
                    x1 > KENAR_PAY
                    and x2 < (FRAME_W - KENAR_PAY)
                    and kutu_yuksekligi_px >= MIN_SAYIM_KUTU_YUKSEKLIK
                )

                if guvenilir_kutu:
                    yeni_durum = "yukari" if merkez_y < CIZGI_Y else "asagi"
                    onceki_durum = durumlar.get(track_id)

                    if onceki_durum is not None and onceki_durum != yeni_durum:
                        bilgi = kisi_bilgisi.get(track_id, {"yas_grubu": "Bilinmiyor", "cinsiyet": "Bilinmiyor"})

                        kayit = {
                            "zaman": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "yas_grubu": bilgi["yas_grubu"],
                            "cinsiyet": bilgi["cinsiyet"],
                            "boy_m": round(kisi_boyu_m, 2) if kisi_boyu_m is not None else None,
                        }

                        if onceki_durum == "yukari" and yeni_durum == "asagi":
                            giris_sayisi += 1
                            kayit["tip"] = "Giris"
                            with gecis_kilidi:
                                gecis_kayitlari.append(kayit)
                        elif onceki_durum == "asagi" and yeni_durum == "yukari":
                            cikis_sayisi += 1
                            kayit["tip"] = "Cikis"
                            with gecis_kilidi:
                                gecis_kayitlari.append(kayit)

                    durumlar[track_id] = yeni_durum
                # guvenilmez kutu -> durum degistirilmez, sayim yapilmaz

                # ---- ETIKET CIZIMI ----
                etiket = f"ID: {track_id}"
                if track_id in kisi_bilgisi:
                    bilgi = kisi_bilgisi[track_id]
                    etiket += f" | {bilgi['cinsiyet']} | {bilgi['yas_grubu']}"
                if kisi_boyu_m is not None:
                    etiket += f" | {kisi_boyu_m:.2f}m"

                cv2.putText(annotated_frame, etiket, (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        with son_kutu_kilidi:
            son_kutu_yukseklikleri.clear()
            son_kutu_yukseklikleri.update(guncel_yukseklikler)

        cv2.putText(annotated_frame, f"Giris: {giris_sayisi}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Cikis: {cikis_sayisi}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # ---- KUCUK px/m ETIKETI ----
        with kalibrasyon_kilidi:
            px_m_metni = f"{pixels_per_meter:.0f}px/m" if is_calibrated else "--px/m"
        cv2.putText(annotated_frame, px_m_metni, (20, annotated_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        with frame_kilidi:
            son_frame = annotated_frame.copy()

        # ---- YEREL ONIZLEME PENCERESI: 'c' ile kalibrasyon, 'q' ile cikis ----
        try:
            cv2.imshow("Kisi Sayma - Kalibrasyon icin 'c' (pencere odakta olmali), cikis icin 'q'", annotated_frame)
            tus = cv2.waitKey(1) & 0xFF
            if tus == ord('c'):
                kalibrasyonu_uygula()
            elif tus == ord('q'):
                break
        except cv2.error:
            # Ekransiz (headless) ortamda calisiyorsa pencere acilamaz;
            # kalibrasyon bu durumda terminalden 'c' + Enter ile yapilabilir.
            pass

    cap.release()
    cv2.destroyAllWindows()


# ==========================================================
# FLASK WEB ARAYUZU (sadece izleme icin)
# ==========================================================
app = Flask(__name__)


def frame_uret():
    while True:
        with frame_kilidi:
            if son_frame is None:
                continue
            basarili, buffer = cv2.imencode('.jpg', son_frame)
            if not basarili:
                continue
            frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video')
def video():
    return Response(frame_uret(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def anasayfa():
    return """
    <html>
    <head><title>Kisi Sayma</title></head>
    <body style="text-align:center; font-family:sans-serif;">
        <h1>Kisi Sayma Sistemi</h1>
        <img src="/video" width="640" height="480">
        <p>Boy kalibrasyonu: onizleme penceresi odaktayken 'c' tusuna basin,
           ya da programi calistirdiginiz terminale 'c' yazip Enter'a basin.</p>
    </body>
    </html>
    """


if __name__ == '__main__':
    kamera_thread = threading.Thread(target=kamera_dongusu, daemon=True)
    kamera_thread.start()

    rapor_thread = threading.Thread(target=aylik_rapor_zamanlayici, daemon=True)
    rapor_thread.start()

    klavye_thread = threading.Thread(target=klavye_dinleyici, daemon=True)
    klavye_thread.start()

    app.run(host='0.0.0.0', port=5000)