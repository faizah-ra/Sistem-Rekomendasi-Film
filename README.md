
# 🎬 Movie Recommendation System using Hybrid Filtering
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)


Sistem rekomendasi film ini dirancang untuk membantu pengguna menemukan tontonan yang relevan dan personal berdasarkan **preferensi konten** maupun **pola perilaku pengguna lain**. Proyek ini dibangun menggunakan pendekatan **Content-Based Filtering (CBF)** dan **Collaborative Filtering (CF)** berbasis *neural network*.  

---

## 💼 Latar Belakang Bisnis

Dalam era digital yang sarat akan konten hiburan, pengguna platform streaming seperti Netflix, Disney+, dan Amazon Prime sering kali mengalami information overload akibat ribuan pilihan film yang tersedia. Tanpa sistem rekomendasi yang efektif, pengguna bisa merasa kewalahan, mengakibatkan:
  - Tingginya bounce rate karena ketidaksesuaian konten,
  - Rendahnya retention rate dan waktu tonton,
  - Menurunnya kepuasan pengguna akibat tidak adanya personalisasi konten.
Untuk itu, sistem rekomendasi menjadi solusi penting dalam menyederhanakan proses pemilihan film dan meningkatkan pengalaman pengguna secara personal.



---

## 🎯 Tujuan Proyek
Proyek ini dikembangkan untuk membangun sistem rekomendasi film yang:
  - Memberikan rekomendasi berbasis kemiripan konten film menggunakan TF-IDF dan Cosine Similarity (Content-Based Filtering / CBF),
  - Memberikan rekomendasi berbasis interaksi pengguna melalui Neural Collaborative Filtering menggunakan arsitektur RecommenderNet (Collaborative Filtering / CF),
  - Mengukur performa kedua pendekatan menggunakan metrik evaluasi seperti Precision@K, MAP@K, RMSE, dan MAE.
Solusi ini juga dirancang agar dapat dengan mudah diintegrasikan ke dalam sistem OTT (Over-The-Top) sebagai modul rekomendasi adaptif, dengan potensi pengembangan menuju pendekatan context-aware dan real-time recommendation.


---

## 🛠️ Teknologi yang Digunakan

- **Python**, Pandas, NumPy, Scikit-Learn
- **Keras** (Tensorflow) untuk model neural network
- **Matplotlib & Seaborn** untuk visualisasi
- Dataset: [MovieLens Small Dataset](https://www.kaggle.com/datasets/shubhammehta21/movie-lens-small-latest-dataset)

---

## 📁 Struktur Repository

```
sistem-rekomendasi-film/
├── assets/                              # Folder berisi aset gambar dan visual pendukung
├── LICENSE                              # Lisensi proyek (MIT)
├── README.md                            # Dokumentasi utama
├── laporan_proyek.md                    # Laporan lengkap proyek
├── requirements.txt                     # Dependensi proyek
├── sistem_rekomendasi_film.ipynb        # Notebook utama (EDA + Modeling)
└── sistem_rekomendasi_film.py           # Versi Python script (.py)
```

---
## 📈 Hasil Evaluasi

| Model | Precision@5 | MAP@5 | RMSE | MAE |
|-------|-------------|-------|------|-----|
| CBF   | 0.1600      | 0.0991| -    | -   |
| CF    | -           | -     | 0.9270| 0.7108 |

---
## 💡 Insight Bisnis
  - Pendekatan hybrid antara CBF dan CF terbukti mampu menangani permasalahan cold-start user maupun sparse rating matrix dengan baik.
  - Model CF berbasis neural network menawarkan potensi personalisasi yang lebih tinggi seiring bertambahnya data interaksi pengguna.
  - CBF tetap relevan untuk pengguna baru atau konten baru tanpa histori rating.
  - Kombinasi kedua pendekatan ini dapat dikembangkan menjadi sistem rekomendasi real-time yang adaptif dengan bantuan data kontekstual (lokasi, waktu, device).
  - Sistem ini berpeluang besar diintegrasikan sebagai komponen rekomendasi dalam platform streaming, e-commerce, hingga aplikasi edukasi berbasis video.
---

## 🚀 Cara Menjalankan

1. Clone repositori ini  
2. Install dependency:  
```bash
pip install -r requirements.txt
```
3. Jalankan kode dari `notebooks/`

---

## 🏆 Pengakuan Proyek
Proyek ini merupakan bagian dari Submission Proyek Akhir Kelas Machine Learning Terapan yang diselenggarakan oleh Dicoding Indonesia, dalam program Coding Camp 2025 powered by DBS Foundation.

✅ Proyek ini telah dinyatakan lulus dengan sempurna (rating lima bintang ⭐⭐⭐⭐⭐) oleh reviewer resmi Dicoding, yang menyatakan bahwa:

"Proyek Sistem Rekomendasi yang kamu kerjakan adalah proyek yang sangat menarik! Kamu telah menerapkannya dengan sangat baik. Kamu memiliki pemahaman yang baik mengenai permasalahan, dataset, tujuan proyek, metode, dan model machine learning. Kamu juga mampu mengomunikasikan seluruh proyek dengan baik. Good job!"
— Reviewer Dicoding

---

## 👩‍💻 Tentang Pengembang

**Faizah Rizki Auliawati**  
Mahasiswa Informatika sekaligus seorang Data Enthusiast dengan minat mendalam pada bidang Machine Learning, System Analysis, dan pengembangan solusi berbasis data. Proyek ini merupakan bagian dari portofolio ilmiah dan praktikal untuk pengembangan sistem cerdas berbasis rekomendasi.

---

## 📄 Lisensi

Proyek ini berlisensi MIT. Lihat `LICENSE` untuk detail.
