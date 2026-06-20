#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script Preprocessing & Sanity Check untuk Deteksi Sampah Daur Ulang (YOLOv8)
-------------------------------------------------------------------------
Script ini dirancang untuk:
1. Memverifikasi struktur folder dataset (train, valid, test).
2. Memperbaiki file `data.yaml` secara otomatis dengan menggunakan path absolut 
   agar program training dan evaluasi dapat dijalankan dari folder manapun tanpa error.
3. Melakukan pemeriksaan integritas dataset (sanity check) untuk memastikan file gambar
   dan file label sinkron.
4. Menghitung statistik sebaran kelas sampah (kaca, kertas, logam, plastik) dalam dataset.
"""

import os
import yaml
import glob
from collections import Counter
import sys

# Konfigurasi UTF-8 untuk output terminal agar mendukung emoji di Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Untuk Python versi lama yang tidak mendukung reconfigure
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def check_and_fix_dataset():
    # 1. Tentukan path root proyek (folder tempat script ini berada, satu tingkat di atas Script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    print("=" * 60)
    # Teks ramah pemula
    print("🔍 SISTEM DETEKSI DAN VALIDASI DATASET (PREPROCESSING)")
    print("=" * 60)
    print(f"Direktori Proyek Terdeteksi: {project_root}")
    
    # 2. Definisikan folder dataset utama
    splits = ['train', 'valid', 'test']
    dataset_status = True
    
    # Cek apakah folder train, valid, test ada di root proyek
    for split in splits:
        split_path = os.path.join(project_root, split)
        if not os.path.exists(split_path):
            print(f"❌ Error: Folder '{split}' TIDAK ditemukan di {project_root}")
            dataset_status = False
        else:
            img_path = os.path.join(split_path, 'images')
            lbl_path = os.path.join(split_path, 'labels')
            if not os.path.exists(img_path) or not os.path.exists(lbl_path):
                print(f"❌ Error: Subfolder 'images' atau 'labels' di dalam '{split}' tidak lengkap!")
                dataset_status = False
            else:
                print(f"✅ Folder '{split}' terverifikasi dengan lengkap.")
                
    if not dataset_status:
        print("\n⚠️ PERINGATAN: Dataset tidak lengkap atau struktur folder salah.")
        print("Pastikan Anda telah mengekstrak dataset zip ke dalam folder proyek ini.")
        return False

    # 3. Validasi dan Update data.yaml secara dinamis
    yaml_path = os.path.join(project_root, "data.yaml")
    if not os.path.exists(yaml_path):
        print(f"⚠️ data.yaml tidak ditemukan di {project_root}. Membuat data.yaml baru...")
        data_yaml = {
            'path': project_root.replace('\\', '/'), # Menggunakan forward slash agar kompatibel
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': 4,
            'names': ['kaca', 'kertas', 'logam', 'plastik']
        }
    else:
        print("📝 Membaca data.yaml lama dan mengupdate path...")
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data_yaml = yaml.safe_load(f)
        except Exception as e:
            print(f"❌ Gagal membaca data.yaml: {e}")
            return False
        
        # Update path ke format absolut yang dinamis
        data_yaml['path'] = project_root.replace('\\', '/')
        data_yaml['train'] = 'train/images'
        data_yaml['val'] = 'valid/images'
        data_yaml['test'] = 'test/images'
        
    # Tulis ulang data.yaml dengan path terupdate
    try:
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
        print(f"✅ data.yaml berhasil diperbarui! Path absolut diset ke: {data_yaml['path']}")
    except Exception as e:
        print(f"❌ Gagal memperbarui data.yaml: {e}")
        return False
        
    # 4. Hitung statistik dataset
    print("\n📊 MENGHITUNG STATISTIK DATASET...")
    class_names = data_yaml.get('names', ['kaca', 'kertas', 'logam', 'plastik'])
    
    for split in splits:
        images_dir = os.path.join(project_root, split, 'images')
        labels_dir = os.path.join(project_root, split, 'labels')
        
        image_files = glob.glob(os.path.join(images_dir, "*"))
        # Filter hanya format gambar yang valid
        image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
        
        print(f"\n📂 Split [{split.upper()}]:")
        print(f"   - Jumlah Gambar: {len(image_files)}")
        print(f"   - Jumlah File Label: {len(label_files)}")
        
        # Sanity check: apakah jumlah gambar dan label sama?
        if len(image_files) != len(label_files):
            print(f"   ⚠️ PERINGATAN: Jumlah gambar ({len(image_files)}) dan label ({len(label_files)}) tidak cocok!")
            
        # Hitung distribusi kelas
        classes_counter = Counter()
        for lbl_file in label_files:
            try:
                with open(lbl_file, 'r') as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            classes_counter[class_id] += 1
            except Exception as e:
                # Lewati jika ada file kosong atau rusak
                continue
                
        print("   - Distribusi Objek Teranotasi:")
        total_objects = sum(classes_counter.values())
        for class_id in sorted(classes_counter.keys()):
            if class_id < len(class_names):
                c_name = class_names[class_id]
            else:
                c_name = f"Unknown-ID-{class_id}"
            count = classes_counter[class_id]
            pct = (count / total_objects * 100) if total_objects > 0 else 0
            print(f"     * {c_name.capitalize()}: {count} objek ({pct:.2f}%)")
        print(f"   - Total Objek Sampah Terdeteksi: {total_objects}")
        if len(image_files) > 0:
            print(f"   - Rata-rata Objek per Gambar: {total_objects / len(image_files):.2f}")
            
    print("\n🎉 Preprocessing dan Validasi Dataset Selesai dengan Sukses!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    check_and_fix_dataset()
