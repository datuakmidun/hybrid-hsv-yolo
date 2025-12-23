import cv2
from ultralytics import YOLO

# 1. Load Model
# Pastikan file model ada di folder yang sama dengan script ini
try:
    model = YOLO("models/orange_ball_yolov8_best.pt")
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Error memuat model: {e}")
    exit()

# 2. Buka Webcam
# Angka 0 biasanya adalah webcam default laptop.
# Jika Anda menggunakan kamera eksternal (USB), coba ganti jadi 1 atau 2.
cap = cv2.VideoCapture(0)

# Cek apakah kamera terbuka
if not cap.isOpened():
    print("Error: Tidak bisa membuka kamera.")
    exit()

# Atur ukuran jendela tampilan (Opsional, biar tidak terlalu besar)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Tekan tombol 'q' di keyboard untuk keluar.")

while True:
    # 3. Baca frame dari kamera
    success, frame = cap.read()
    
    if not success:
        print("Gagal membaca frame kamera.")
        break

    # 4. Lakukan Deteksi (Inference)
    # conf=0.5 artinya hanya deteksi dengan keyakinan > 50% yang ditampilkan
    results = model(frame, conf=0.5, verbose=False)

    # 5. Visualisasi
    # results[0].plot() otomatis menggambar kotak dan label di gambar
    annotated_frame = results[0].plot()

    # 6. Tampilkan Hasil
    cv2.imshow("Deteksi Bola Oranye (YOLOv8)", annotated_frame)

    # 7. Keluar jika tekan 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan memori dan tutup kamera
cap.release()
cv2.destroyAllWindows()