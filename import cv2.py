import cv2
import numpy as np

# Kamera başlatma (Atölyedeki ilk testler için 0)
cap = cv2.VideoCapture(0)

# --- KALİBRASYON DEĞERLERİ ---
# Bu değerleri sahnenize göre ayarlamanız gerekir.
# Örnek: Sahneye koyduğunuz bilinen bir nesnenin piksel alanı veya piksel-metre oranı.
# Basit oran için: 1 metrenin ekranda kaç piksele denk geldiğini bulun.
PIXELS_PER_METER = 100.0  # Örnek varsayım: 1 metre = 100 piksel (Kameraya uzaklığa göre değişir)

print("Kamera açıldı. Çıkmak için 'q' tuşuna basın.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Görüntüyü gri tonlamaya çevir ve bulanıklaştır (gürültüyü azaltmak için)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Eşikleme (Threshold) veya Kenar Tespiti ile nesneyi/bölgeyi belirginleştir
    # Burada basitçe bir eşikleme yapıyoruz (üzerinde çalışacağınız nesneye göre ayarlanmalıdır)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

    # 3. Konturları (sınırları) bulma
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Çok küçük gürültüleri elemek için alan filtresi
        area_pixels = cv2.contourArea(cnt)
        if area_pixels > 1000:  # Eşik değer
            
            # Nesnenin etrafına sınırlayıcı kutu veya çember çizelim
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 4. Piksel Alanını Metrekareye Çevirme
            # Alan (piksel^2) / (Piksel/Metre)^2 = Alan (m^2)
            area_m2 = area_pixels / (PIXELS_PER_METER ** 2)

            # Ekrana sonuçları yazdırma
            text = f"Alan: {area_m2:.4f} m2"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Görüntüyü göster
    cv2.imshow("Metrekare Hesaplama Testi", frame)

    # 'q' tuşuna basılırsa döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()