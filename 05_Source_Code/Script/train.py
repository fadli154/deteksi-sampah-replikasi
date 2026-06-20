#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script Training YOLOv8 untuk Deteksi Sampah Daur Ulang
---------------------------------------------------
Script ini dirancang untuk:
1. Mendeteksi hardware (GPU CUDA vs CPU) secara otomatis.
2. Memuat pre-trained model YOLOv8 (yolov8n.pt atau yolov8s.pt).
3. Melatih model menggunakan konfigurasi dataset dari `data.yaml` secara reproducible.
4. Menyimpan bobot terbaik (*best weights*) hasil training di folder `runs/detect/`.
"""

import os
import torch
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

def train_model(model_type="yolov8n", epochs=60, batch_size=12, imgsz=640, device=None):
    # 1. Tentukan path root proyek
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    data_yaml_path = os.path.join(project_root, "data.yaml")
    
    print("=" * 60)
    print("🚀 MEMULAI PROSES TRAINING YOLOv8")
    print("=" * 60)
    
    # 2. Cek Hardware GPU CUDA
    print("🔍 Memeriksa Hardware...")
    print(f"   - Versi PyTorch: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"   - GPU CUDA Tersedia: {cuda_available}")
    
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"   - GPU Terdeteksi: {gpu_name}")
        if device is None:
            device = 0 # Default menggunakan GPU 0
    else:
        print("   - ⚠️ GPU CUDA tidak terdeteksi. Proses training akan menggunakan CPU (Sangat Lambat!)")
        if device is None:
            device = 'cpu'

    # 3. Validasi Keberadaan data.yaml
    if not os.path.exists(data_yaml_path):
        print(f"❌ Error: File konfigurasi dataset '{data_yaml_path}' tidak ditemukan!")
        print("Silakan jalankan script preprocessing.py terlebih dahulu untuk memperbaikinya.")
        return False

    # 4. Inisialisasi Model YOLOv8
    # Menentukan model name
    model_name = f"{model_type}.pt"
    print(f"\n📥 Memuat Model Pre-trained: {model_name}...")
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"❌ Gagal memuat model {model_name}: {e}")
        return False

    # 5. Menampilkan Ringkasan Parameter Training
    print("\n📋 Ringkasan Parameter Pelatihan:")
    print(f"   - Model Varian: {model_type}")
    print(f"   - Path Konfigurasi Dataset: {data_yaml_path}")
    print(f"   - Jumlah Epochs: {epochs}")
    print(f"   - Ukuran Gambar (Image Size): {imgsz}x{imgsz}")
    print(f"   - Batch Size: {batch_size}")
    print(f"   - Device: {device}")
    print(f"   - Lokasi Output: runs/detect/")
    print("-" * 60)
    print("Pelatihan akan dimulai. Ini mungkin memakan waktu beberapa jam tergantung hardware Anda.")
    print("Silakan tunggu...")
    print("-" * 60)
    
    # 6. Eksekusi Training
    try:
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            workers=4,
            cache=False,
            device=device,
            name=f"train-{model_type}" # Nama folder penyimpanan otomatis
        )
        print("\n🎉 TRAINING SELESAI DENGAN SUKSES!")
        print(f"Hasil training, kurva, dan bobot disimpan di: {results.save_dir}")
        print(f"Bobot terbaik Anda berada di: {os.path.join(results.save_dir, 'weights', 'best.pt')}")
        print("=" * 60)
        return True
    except Exception as e:
        print(f"\n❌ Error terjadi saat proses training: {e}")
        print("Saran perbaikan:")
        print("1. Jika kehabisan memori GPU (Out of Memory), coba kecilkan 'batch_size' (misal ke 8 atau 4).")
        print("2. Pastikan dependensi di requirement.txt sudah terinstal sempurna.")
        print("=" * 60)
        return False

if __name__ == "__main__":
    # Menambahkan argument parser agar mudah dijalankan lewat terminal bagi pengguna advance
    parser = argparse.ArgumentParser(description="Train YOLOv8 model for waste detection.")
    parser.add_argument("--model", type=str, default="yolov8n", choices=["yolov8n", "yolov8s"], 
                        help="Tipe model YOLOv8 (yolov8n atau yolov8s)")
    parser.add_argument("--epochs", type=int, default=60, help="Jumlah epoch pelatihan")
    parser.add_argument("--batch", type=int, default=12, help="Ukuran batch")
    parser.add_argument("--imgsz", type=int, default=640, help="Ukuran gambar input")
    parser.add_argument("--device", type=str, default=None, help="Device untuk training (0, cpu, dll)")
    
    args = parser.parse_args()
    train_model(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device
    )
