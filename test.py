from ultralytics import YOLO
import os

# 1. Muat model hasil training kamu
model = YOLO('runs/detect/train-yolov8n/weights/best.pt')  # ganti path sesuai lokasi file kamu

# 2. Siapkan folder berisi citra uji di luar dataset
test_folder = 'citra_uji_eksternal'  # ganti sesuai folder kamu

# 3. Jalankan deteksi untuk tiap citra dan hitung jumlah objek
for filename in sorted(os.listdir(test_folder)):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', 'webp')):
        path = os.path.join(test_folder, filename)
        results = model.predict(source=path, save=True, conf=0.25)
        
        jumlah_terdeteksi = len(results[0].boxes)
        kelas_terdeteksi = [model.names[int(cls)] for cls in results[0].boxes.cls]
        
        print(f"{filename}: {jumlah_terdeteksi} objek terdeteksi -> {kelas_terdeteksi}")