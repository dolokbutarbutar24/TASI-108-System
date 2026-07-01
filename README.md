# Cassava Leaf Disease Detection System using Fine-Tuned YOLOv12

Sistem deteksi penyakit daun singkong berbasis deep learning menggunakan arsitektur **YOLOv12**, dirancang untuk mengatasi tantangan **occlusion** dan **latar belakang kompleks** pada kondisi lingkungan lahan nyata.

## 👥 Tim Peneliti

| Nama | NIM |
|------|-----|
| Dolok Oktavianus Putra Butarbutar | 12S22009 |
| Ira Wianda Sari Silalahi | 12S22048 |
| Sefanya Yemima Sinaga | 12S22051 |

**Program Studi:** Sarjana Sistem Informasi — Institut Teknologi Del

## 📄 Judul Penelitian

**A Fine-Tuned YOLOv12 System for Cassava Leaf Disease Detection Under Occlusion and Complex Background Conditions**

## 📌 Abstrak

Singkong (*Manihot esculenta*) merupakan komoditas pangan dan industri strategis di Indonesia yang produktivitasnya terancam oleh berbagai penyakit daun, dengan potensi penurunan hasil panen mencapai 95%. Identifikasi manual memiliki tingkat subjektivitas tinggi, sehingga diperlukan solusi otomatis berbasis deep learning.

Penelitian ini mengevaluasi efektivitas arsitektur YOLOv12 dalam mendeteksi penyakit daun singkong pada kondisi lingkungan lahan, dengan fokus utama membandingkan performa **model baseline** dan **model hasil fine-tuning**. Dalam prosesnya, dilakukan perbaikan anotasi (*re-labeling*) dari skala **area gejala daun** menjadi **daun utuh**, setelah ditemukan bahwa pelabelan awal kurang optimal untuk mengenali konteks objek secara luas. Penelitian ini juga menambahkan **data primer** hasil pengambilan langsung di lapangan menggunakan kamera smartphone, yang digabungkan dengan data sekunder untuk pelatihan model final.

Evaluasi terhadap tantangan *occlusion* dan latar belakang kompleks dilakukan melalui pendekatan **Bayesian Optimization** dengan algoritma **TPE (Tree-structured Parzen Estimator)** via **Optuna**.

### Hasil Utama

| Model | mAP50 | mAP50-95 |
|-------|-------|----------|
| YOLOv12s Baseline | 0,825 | 0,622 |
| YOLOv12s Fine-Tuned | **0,834** | **0,623** |

Hasil eksperimen menunjukkan bahwa model YOLOv12s hasil fine-tuning memberikan peningkatan performa pada seluruh metrik akurasi. Penelitian ini menyimpulkan bahwa penggunaan arsitektur YOLOv12 yang dikombinasikan dengan strategi pelabelan objek secara utuh terbukti efektif untuk meningkatkan akurasi dan lokalisasi sistem deteksi penyakit tanaman.

**Kata kunci:** YOLOv12, Penyakit Daun Singkong, Occlusion, Latar Belakang Kompleks, Fine-Tuning, Hyperparameter Optimization, Object Annotation.

---

## 🔬 Desain Eksperimen

Penelitian ini terdiri dari tiga skenario eksperimen utama:

1. **Eksperimen 1** — Anotasi area gejala, menggunakan data Kaggle saja
2. **Eksperimen 2** — Anotasi daun utuh (*whole-leaf*), menggunakan data Kaggle saja
3. **Eksperimen 3** — Anotasi daun utuh, menggunakan data Kaggle + data primer (hasil pengambilan lapangan)

Setiap skenario dievaluasi pada dua konfigurasi model:
- **Baseline** — hyperparameter default YOLOv12
- **Fine-tuned** — hasil Hyperparameter Optimization (HPO) menggunakan Optuna dengan sampler TPE

## 🍃 Kelas Penyakit

Model dilatih untuk mengklasifikasikan 5 kelas:

| Kelas | Keterangan |
|-------|------------|
| CBB | Cassava Bacterial Blight |
| CBSD | Cassava Brown Streak Disease |
| CGM | Cassava Green Mottle |
| CMD | Cassava Mosaic Disease |
| HEALTHY | Daun sehat |

## 📊 Dataset

- **Cassava Leaf Disease Classification** (Kaggle) — [link](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/overview)
- **iCassava 2019 Fine-Grained Visual Categorization Challenge** (Kaggle) — [link](https://www.kaggle.com/competitions/cassava-disease/data)
- **Data primer** — citra daun singkong hasil pengambilan langsung di lapangan menggunakan kamera smartphone
- **Dataset negative sample** (COCO) — [link](https://cocodataset.org/#download)

> Dataset tidak disertakan langsung di repository ini karena ukurannya besar. Silakan unduh dari link Kaggle di atas, lalu gunakan skrip pada folder `training/data_preparation/` untuk melakukan integrasi dan pembagian data.

## ⚙️ Metodologi

- **Arsitektur Model:** YOLOv12s
- **Anotasi:** Manual menggunakan platform [Roboflow](https://roboflow.com/)
- **Hyperparameter Optimization:** Optuna (Bayesian Optimization, algoritma TPE)
- **Pembagian Data:** 70% train, 20% validation, 10% test
- **Metrik Evaluasi:**
  - Mean Average Precision (mAP50, mAP50-95)

## 🖥️ Sistem Deteksi

Model terbaik diintegrasikan ke dalam aplikasi berbasis website menggunakan **Streamlit**, yang memungkinkan pengguna mengunggah atau mengambil foto daun singkong secara langsung dan mendapatkan hasil deteksi penyakit secara real-time, lengkap dengan visualisasi bounding box dan tingkat keyakinan (*confidence score*).

*Repository ini merupakan bagian dari Tugas Akhir Program Studi Sarjana Sistem Informasi, Institut Teknologi Del.*
