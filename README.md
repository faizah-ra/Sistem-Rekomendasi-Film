
# 🎬 Movie Recommendation System using Hybrid Filtering
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)


Sistem rekomendasi film ini dirancang untuk membantu pengguna menemukan tontonan yang relevan dan personal berdasarkan **preferensi konten** maupun **pola perilaku pengguna lain**. Proyek ini dibangun menggunakan pendekatan **Content-Based Filtering (CBF)** dan **Collaborative Filtering (CF)** berbasis *neural network*.  

---

## 🔍 Studi Kasus dan Permasalahan

Dalam ekosistem platform streaming seperti Netflix, Disney+, dan Amazon Prime, pengguna dihadapkan pada ribuan pilihan konten yang dapat menyebabkan information overload. Hal ini menimbulkan tantangan dalam menemukan film yang sesuai preferensi. Sistem rekomendasi ini bertujuan untuk mengurangi beban kognitif pengguna dan meningkatkan pengalaman menonton secara personal.

---

## 🎯 Tujuan Proyek

- Membangun sistem rekomendasi berbasis **TF-IDF + Cosine Similarity** (CBF)
- Membangun model rekomendasi berbasis **Neural Collaborative Filtering** (RecommenderNet)
- Membandingkan hasil kedua pendekatan menggunakan metrik evaluasi seperti Precision@K, MAP@K, RMSE, dan MAE

---

## 🛠️ Teknologi yang Digunakan

- **Python**, Pandas, NumPy, Scikit-Learn
- **Keras** (Tensorflow) untuk model neural network
- **Matplotlib & Seaborn** untuk visualisasi
- Dataset: [MovieLens Small Dataset](https://www.kaggle.com/datasets/shubhammehta21/movie-lens-small-latest-dataset)

---

## 📁 Struktur Repository

```
movie-recommender-portfolio/
├── README.md
├── requirements.txt        # Dependencies
├── notebooks/              # Notebook EDA & Modeling
├── src/                    # Kode modular (preprocessing, modeling)
├── preview/                # Screenshot hasil
├── docs/                   # Diagram DFD, UML, ERD, Mockup, Proposal, Laporan
└── .gitignore              # File yang diabaikan Git
```

---

## 📊 Diagram Sistem & Dokumen Pendukung

Tersedia di folder `docs/`:
- ✅ **DFD Level 0 & 1**: Alur data antar entitas dan proses utama
- ✅ **ERD**: Struktur relasional dari dataset `ratings.csv` dan `movies.csv`
- ✅ **UML Use Case & Sequence**: Interaksi pengguna dan sistem rekomendasi
- ✅ **Proposal & Laporan PDF**: Dokumen ilmiah lengkap

---

## 📈 Hasil Evaluasi

| Model | Precision@5 | MAP@5 | RMSE | MAE |
|-------|-------------|-------|------|-----|
| CBF   | 0.1600      | 0.0991| -    | -   |
| CF    | -           | -     | 0.9270| 0.7108 |

---

## 🚀 Cara Menjalankan

1. Clone repositori ini  
2. Install dependency:  
```bash
pip install -r requirements.txt
```
3. Jalankan kode dari `app.py` atau `notebooks/`

---

## 👩‍💻 Tentang Pengembang

**Faizah Rizki Auliawati**  
Mahasiswa Informatika sekaligus seorang Data Enthusiast dengan minat mendalam pada bidang Machine Learning, System Analysis, dan pengembangan solusi berbasis data. Proyek ini merupakan bagian dari portofolio ilmiah dan praktikal untuk pengembangan sistem cerdas berbasis rekomendasi.

---

## 📄 Lisensi

Proyek ini berlisensi MIT. Lihat `LICENSE` untuk detail.
