#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script Pengujian (Testing) Model YOLOv8 pada Dataset Test
------------------------------------------------------
Script ini digunakan untuk:
1. Memuat model YOLOv8 hasil training yang disimpan di folder `runs/`.
2. Melakukan evaluasi performa model menggunakan dataset uji (*test split*).
3. Mencetak metrik evaluasi formal (Precision, Recall, mAP50, mAP50-95).
4. Menyediakan penjelasan sederhana tentang arti metrik untuk pengguna awam.
"""

import os
import argparse
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

def evaluate_model(weights_path=None, model_type="yolov8n"):
    # 1. Tentukan path root proyek
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    data_yaml_path = os.path.join(project_root, "data.yaml")
    
    print("=" * 60)
    print("📊 PENGUJIAN MODEL YOLOv8 PADA DATASET TEST")
    print("=" * 60)

    # 2. Cari bobot model terbaik jika tidak dispesifikasikan oleh pengguna
    if weights_path is None:
        # Mencari di folder runs default hasil training
        possible_paths = [
            os.path.join(project_root, "runs", "detect", f"train-{model_type}", "weights", "best.pt"),
            os.path.join(project_root, "runs", "detect", f"train-{model_type}", "best.pt"),
            os.path.join(project_root, f"best_{model_type}.pt"),
            os.path.join(project_root, "yolov8n.pt") # Fallback ke pre-trained
        ]
        
        # Tambahan: Cari path training asli dari folder lama di repo
        possible_paths.append(os.path.join(project_root, "runs", "detect", "train-yolov8n", "weights", "best.pt"))
        possible_paths.append(os.path.join(project_root, "runs", "detect", "train-yolov8s", "weights", "best.pt"))
        
        for path in possible_paths:
            if os.path.exists(path):
                weights_path = path
                break
                
    if weights_path is None or not os.path.exists(weights_path):
        print("❌ Error: Bobot model (best.pt) tidak ditemukan!")
        print("Pastikan Anda sudah menjalankan training terlebih dahulu atau menempatkan file 'best.pt' pada folder yang sesuai.")
        return False
        
    print(f"✅ Menggunakan Bobot Model: {weights_path}")
    
    # 3. Muat Model
    try:
        model = YOLO(weights_path)
    except Exception as e:
        print(f"❌ Gagal memuat model dari {weights_path}: {e}")
        return False
        
    # 4. Validasi data.yaml
    if not os.path.exists(data_yaml_path):
        print(f"❌ Error: data.yaml tidak ditemukan di {data_yaml_path}")
        return False

    # 5. Jalankan Evaluasi pada split='test'
    print("\n⏳ Mengevaluasi model pada dataset pengujian (test set)...")
    try:
        metrics = model.val(
            data=data_yaml_path,
            split='test',
            plots=True
        )
        
        # 6. Tampilkan Hasil Metrik
        precision = metrics.box.mp     # Mean Precision
        recall = metrics.box.mr        # Mean Recall
        map50 = metrics.box.map50      # mAP at IoU=0.50
        map50_95 = metrics.box.map     # mAP at IoU=0.50:0.95
        
        print("\n" + "=" * 50)
        print("🏆 HASIL EVALUASI MODEL:")
        print("=" * 50)
        print(f"   📈 Precision (Presisi) : {precision:.4f} ({precision * 100:.1f}%)")
        print(f"   📈 Recall (Daya Ingat) : {recall:.4f} ({recall * 100:.1f}%)")
        print(f"   📈 mAP50                : {map50:.4f} ({map50 * 100:.1f}%)")
        print(f"   📈 mAP50-95             : {map50_95:.4f} ({map50_95 * 100:.1f}%)")
        print("=" * 50)
        
        # 7. Penjelasan istilah bagi orang awam
        print("\n💡 APA ARTI ANGKA-ANGKA INI BAGI PEMULA?")
        print("- Precision (Presisi): Seberapa akurat tebakan model. Jika bernilai 95%, berarti dari 100")
        print("  objek yang dideteksi model sebagai sampah daur ulang, 95 di antaranya benar sampah tersebut.")
        print("- Recall (Daya Ingat): Seberapa banyak sampah asli yang berhasil ditemukan. Jika bernilai 88%,")
        print("  artinya model berhasil mendeteksi 88 dari 100 sampah asli yang ada di dalam gambar.")
        print("- mAP50 (Mean Average Precision): Tolok ukur akurasi keseluruhan model pada overlap deteksi 50%.")
        print("  Semakin mendekati 1.0 (100%), semakin pintar dan presisi model tersebut!")
        print("=" * 50)
        
        return True
    except Exception as e:
        print(f"❌ Terjadi kesalahan saat evaluasi: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model on test split.")
    parser.add_argument("--weights", type=str, default=None, help="Path ke file bobot model (.pt)")
    parser.add_argument("--model", type=str, default="yolov8n", choices=["yolov8n", "yolov8s"], 
                        help="Tipe model default untuk dicari")
    
    args = parser.parse_args()
    evaluate_model(weights_path=args.weights, model_type=args.model)
