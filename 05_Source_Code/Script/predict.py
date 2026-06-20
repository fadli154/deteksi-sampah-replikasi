#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script Inferensi/Prediksi YOLOv8 untuk Citra Uji
-----------------------------------------------
Script ini digunakan untuk:
1. Memuat model YOLOv8 (format PyTorch `.pt` atau TensorFlow Lite `.tflite`).
2. Menjalankan deteksi pada citra uji baru (misal folder `citra_uji_eksternal`).
3. Menghitung jumlah objek sampah yang terdeteksi beserta jenis kategorinya.
4. Menyimpan citra hasil deteksi lengkap dengan bounding box di `runs/detect/predict/`.
"""

import os
import argparse
import glob
from ultralytics import YOLO
import sys

# Konfigurasi UTF-8 untuk output terminal agar mendukung emoji di Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def run_prediction(source_path=None, weights_path=None, conf_threshold=0.25):
    # 1. Tentukan path root proyek
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    print("=" * 60)
    print("🔮 DETEKSI SAMPAH MENGGUNAKAN MODEL YOLOv8")
    print("=" * 60)

    # 2. Cari model terbaik secara otomatis jika tidak disediakan
    if weights_path is None:
        possible_paths = [
            os.path.join(project_root, "runs", "detect", "train-yolov8n", "weights", "best.pt"),
            os.path.join(project_root, "runs", "detect", "train-yolov8s", "weights", "best.pt"),
            os.path.join(project_root, "runs", "detect", "train-yolov8n", "weights", "best_yolov8n.tflite"),
            os.path.join(project_root, "runs", "detect", "train-yolov8s", "weights", "best_yolov8s.tflite"),
            os.path.join(project_root, "yolov8n.pt")
        ]
        for path in possible_paths:
            if os.path.exists(path):
                weights_path = path
                break

    if weights_path is None or not os.path.exists(weights_path):
        print("❌ Error: File bobot model (best.pt atau best_yolov8n.tflite) tidak ditemukan!")
        print("Pastikan model sudah dilatih atau diletakkan di folder proyek.")
        return False

    print(f"✅ Memuat Model: {weights_path}")
    try:
        model = YOLO(weights_path)
    except Exception as e:
        print(f"❌ Gagal memuat model: {e}")
        return False

    # 3. Tentukan input citra uji
    if source_path is None:
        source_path = os.path.join(project_root, "citra_uji_eksternal")
        
    if not os.path.exists(source_path):
        print(f"❌ Error: Folder/File input '{source_path}' tidak ditemukan!")
        return False

    # 4. Deteksi apakah input berupa file tunggal atau direktori
    if os.path.isdir(source_path):
        # Ambil semua file gambar di dalam folder
        extensions = ('*.jpg', '*.jpeg', '*.png', '*.webp')
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(source_path, ext)))
        # Sortir nama file agar rapi
        image_files.sort()
    else:
        image_files = [source_path]

    if not image_files:
        print(f"⚠️ Peringatan: Tidak ada file gambar (.jpg, .png, .webp) ditemukan di {source_path}")
        return False

    print(f"📂 Menemukan {len(image_files)} gambar untuk diuji.")
    print(f"🚀 Menjalankan deteksi (Confidence Threshold: {conf_threshold})...\n")

    # 5. Jalankan inferensi untuk setiap citra
    success_count = 0
    for img_path in image_files:
        filename = os.path.basename(img_path)
        try:
            # Jalankan prediksi dan simpan visualisasi gambar
            # save=True akan menyimpan gambar ber-bounding box secara otomatis
            results = model.predict(source=img_path, save=True, conf=conf_threshold, verbose=False)
            
            # Ambil kotak koordinat dan label kelas yang terdeteksi
            boxes = results[0].boxes
            jumlah_terdeteksi = len(boxes)
            
            # Ambil nama kelas yang terdeteksi
            kelas_terdeteksi = []
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                kelas_terdeteksi.append(class_name)
                
            print(f"📸 {filename}:")
            print(f"   -> Terdeteksi {jumlah_terdeteksi} objek sampah.")
            if jumlah_terdeteksi > 0:
                # Hitung jumlah per kelas
                breakdown = {}
                for cls in kelas_terdeteksi:
                    breakdown[cls] = breakdown.get(cls, 0) + 1
                
                breakdown_str = ", ".join([f"{k}: {v}" for k, v in breakdown.items()])
                print(f"   -> Rincian: {breakdown_str}")
            else:
                print("   -> (Tidak ada sampah daur ulang terdeteksi)")
            
            success_count += 1
        except Exception as e:
            print(f"❌ Gagal memproses gambar {filename}: {e}")

    print("\n" + "=" * 60)
    print(f"🎉 SELESAI! Berhasil memproses {success_count}/{len(image_files)} gambar.")
    print("Hasil gambar ber-bounding box disimpan di folder:")
    print("📁 runs/detect/predict/ (atau predict2, predict3, dst.)")
    print("=" * 60)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using trained YOLOv8 model.")
    parser.add_argument("--source", type=str, default=None, help="Path ke file gambar atau folder citra uji")
    parser.add_argument("--weights", type=str, default=None, help="Path ke file bobot (.pt atau .tflite)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    
    args = parser.parse_args()
    run_prediction(source_path=args.source, weights_path=args.weights, conf_threshold=args.conf)
