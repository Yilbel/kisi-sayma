import cv2
import numpy as np
from ultralytics import YOLO

# --- 1. TEMEL AYARLAR ---
CAMERA_SOURCE = 0
LINE_Y = 300
KNOWN_USER_HEIGHT_M = 1.75  # Kendi boyunuz (metre)

# Kalibrasyon Durum Değişkenleri
pixels_per_meter = 100.0
is_calibrated = False

# YOLOv8 Modelini Yükle
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(CAMERA_SOURCE)

# Sayaçlar
giris_sayisi = 0
cikis_sayisi = 0
gecmis_idler = set()

print("--- OPTİMİZE EDİLMİŞ SENARYO A ---")
print("1. Çizgi hizasına geçin.")
print("2. Boyunuzu kaydetmek için klavyeden 'c' tuşuna basın.")
print("Çıkmak için 'q' tuşuna basın.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # YOLOv8 ile Nesne Tespiti ve Takibi
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            box_height_pixels = y2 - y1  # Kişinin piksel cinsinden boyu
            merkez_y = int((y1 + y2) / 2)

            # Boy Hesaplama
            kisi_boyu_m = 0.0
            if is_calibrated and pixels_per_meter > 0:
                kisi_boyu_m = box_height_pixels / pixels_per_meter

            # Çizimler
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            if is_calibrated:
                etiket = f"ID: {int(track_id)} | Boy: {kisi_boyu_m:.2f}m"
            else:
                etiket = f"ID: {int(track_id)} (Kalib. Bekliyor)"

            cv2.putText(frame, etiket, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Çizgi Geçiş Kontrolü
            if abs(merkez_y - LINE_Y) < 10 and int(track_id) not in gecmis_idler:
                gecmis_idler.add(int(track_id))
                giris_sayisi += 1

    # --- ARAYÜZ BİLGİLERİ ---
    cv2.line(frame, (0, LINE_Y), (w, LINE_Y), (0, 255, 255), 2)
    cv2.putText(frame, f"Giris: {giris_sayisi} | Cikis: {cikis_sayisi}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    kalib_durum = f"Kalibrasyon: Aktif ({pixels_per_meter:.1f} px/m)" if is_calibrated else "Kalibrasyon: Bekleniyor ('c' tusuna basin)"
    cv2.putText(frame, kalib_durum, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    cv2.imshow("Park Kamera - Optimize Senaryo A", frame)

    # --- KRİTİK OPTİMİZASYON: TUŞ KONTROLÜ DÖNGÜ SONUNDA ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        # 'c' tuşuna basıldığında o an ekrandaki ilk kişinin boyu referans alınır
        if results[0].boxes.id is not None and len(results[0].boxes.xyxy) > 0:
            first_box = results[0].boxes.xyxy[0].cpu().numpy()
            f_height = first_box[3] - first_box[1]
            pixels_per_meter = f_height / KNOWN_USER_HEIGHT_M
            is_calibrated = True
            print(f"[BAŞARILI] Kalibrasyon tamamlandı! 1 Metre = {pixels_per_meter:.2f} piksel olarak kilitlendi.")
        else:
            print("[UYARI] Kalibrasyon için kameranın önünde görünür bir kişi olmalıdır!")

cap.release()
cv2.destroyAllWindows()