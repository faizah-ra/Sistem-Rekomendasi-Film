# LAPORAN PROYEK SISTEM REKOMENDASI FILM BERBASIS MACHINE LEARNING 
Disusun oleh: Faizah Rizki Auliawati

## Project Overview

Di era digital saat ini, layanan streaming seperti **Netflix**, **Amazon Prime Video**, dan **Disney+** menyajikan ribuan film dan serial yang terus bertambah setiap hari. Kelimpahan konten ini justru menimbulkan tantangan berupa *information overload*, yaitu kondisi ketika pengguna kesulitan memilih tontonan yang sesuai dengan preferensinya. Fenomena ini berdampak pada penurunan kepuasan pengguna, waktu menonton yang lebih singkat, serta meningkatnya kemungkinan pengguna meninggalkan platform (*churn*). Menurut penelitian yang dipublikasikan oleh Springer, pengguna sering mengalami kebingungan akibat terlalu banyak pilihan yang tersedia [(Zhou et al., 2010)](https://doi.org/10.1073/pnas.1000488107).

Untuk mengatasi permasalahan tersebut, sistem rekomendasi menjadi solusi penting dalam menyaring informasi dan menyajikan konten yang relevan bagi pengguna. Menurut laporan dari IBM, sistem rekomendasi mampu meningkatkan pengalaman pengguna, loyalitas pelanggan, serta konversi bisnis, sehingga menjadi komponen kunci dalam ekosistem digital modern, khususnya pada platform Over-The-Top (OTT) [(IBM, 2021)](https://www.ibm.com/downloads/cas/EXK4XKX8).

Terdapat dua pendekatan utama yang umum digunakan dalam sistem rekomendasi, yaitu **Content-Based Filtering (CBF)** dan **Collaborative Filtering (CF)**. CBF memberikan rekomendasi berdasarkan karakteristik konten, seperti genre, aktor, atau sutradara, untuk mencocokkan film dengan preferensi pengguna sebelumnya. Pendekatan ini efektif untuk personalisasi awal karena mampu memahami kebutuhan individual, namun cenderung menghasilkan rekomendasi yang terlalu mirip, sehingga kurang memberikan keberagaman [(Lops et al., 2011)](https://doi.org/10.1007/978-0-387-85820-3_5).

Di sisi lain, CF menganalisis pola interaksi antar pengguna untuk menemukan hubungan tersembunyi antara preferensi pengguna yang serupa. Salah satu pendekatan modern dalam kategori ini adalah **Neural Collaborative Filtering (NCF)**, yang menggunakan jaringan saraf untuk mempelajari interaksi non-linear antara pengguna dan item. Menurut penelitian dari [(He et al., 2017)](https://doi.org/10.1145/3038912.3052569), NCF mampu menangkap hubungan kompleks dalam data interaksi dan menghasilkan prediksi yang lebih akurat dibandingkan pendekatan klasik berbasis matriks.

Namun demikian, masing-masing pendekatan memiliki keterbatasan. CBF cenderung terbatas pada item yang serupa dengan riwayat pengguna sebelumnya (*serendipity problem*), sementara CF rentan terhadap *cold start problem*, yaitu kesulitan memberikan rekomendasi kepada pengguna atau item baru yang belum memiliki cukup data historis [(Bobadilla et al., 2013)](https://doi.org/10.1016/j.knosys.2012.11.017).

Sebagai solusi terhadap keterbatasan tersebut, pendekatan **hybrid** yang menggabungkan CBF dan CF menjadi alternatif yang lebih efektif. Dengan mengombinasikan kekuatan analisis konten dan pola interaksi pengguna, sistem rekomendasi hybrid mampu menghasilkan rekomendasi yang lebih **relevan, beragam**, dan **adaptif** terhadap kebutuhan pengguna. Menurut studi oleh [(Zhang et al., 2019)](https://doi.org/10.1016/j.inffus.2018.01.007), pendekatan hybrid tidak hanya meningkatkan akurasi prediksi, tetapi juga meningkatkan keterlibatan (*engagement*) pengguna dan kepuasan pengalaman menonton.

Berdasarkan pertimbangan tersebut, proyek ini dirancang untuk membangun dan membandingkan dua sistem rekomendasi—Content-Based Filtering dan Collaborative Filtering berbasis neural network (RecommenderNet). Diharapkan, sistem ini dapat memberikan rekomendasi film yang personal, efisien, dan relevan bagi pengguna layanan streaming digital.


## Business Understanding

### Problem Statements
Masalah-masalah utama yang diidentifikasi dalam konteks ini meliputi:
- Bagaimana menyarankan film yang relevan berdasarkan preferensi pengguna, terutama dari konten seperti genre yang disukai?
- Bagaimana memberikan rekomendasi yang personal dan akurat untuk pengguna baru (cold-start), yang belum memiliki banyak riwayat interaksi?
- Pendekatan mana yang memberikan performa lebih baik dalam konteks sistem rekomendasi: Content-Based Filtering (CBF) atau Collaborative Filtering (CF)?

### Goals
Sebagai upaya untuk mengatasi tantangan di atas, proyek ini menetapkan beberapa tujuan berikut:
- Membangun sistem rekomendasi berbasis konten (CBF) menggunakan informasi dari metadata film dan menghitung kemiripan antar film dengan teknik TF-IDF dan Cosine Similarity.
- Membangun sistem Collaborative Filtering (CF) berbasis neural network (RecommenderNet) dengan embedding layer untuk memahami preferensi pengguna melalui pola interaksi historis.
- Mengevaluasi dan membandingkan kinerja kedua metode (CBF dan CF) menggunakan metrik seperti Precision@K, MAP@K, RMSE, dan MAE, guna menilai efektivitas dan kualitas rekomendasi yang dihasilkan.

### Solution Statements

Untuk merealisasikan tujuan yang telah dirumuskan, dua pendekatan utama diimplementasikan dan dibandingkan dalam proyek ini:

##### 1. Content-Based Filtering (CBF)
- **Teknologi yang digunakan:** TF-IDF + Cosine Similarity  
- **Sumber data:** Metadata film (genre, judul, deskripsi)  
- **Keunggulan:** Tidak bergantung pada pengguna lain dan cocok untuk pengguna baru  
- **Tujuan:** Memberikan rekomendasi berdasarkan kesamaan konten dari film yang disukai sebelumnya

##### 2. Collaborative Filtering (CF) dengan RecommenderNet
- **Teknologi yang digunakan:** Neural Network dengan Embedding Layer  
- **Sumber data:** Riwayat interaksi antara pengguna dan film  
- **Keunggulan:** Dapat mengidentifikasi pola kompleks dan memberikan rekomendasi personal secara adaptif  
- **Tujuan:** Memberikan rekomendasi berdasarkan kemiripan pola perilaku antar pengguna
  

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah [MovieLens Small Latest Dataset](https://www.kaggle.com/datasets/shubhammehta21/movie-lens-small-latest-dataset) yang tersedia di platform Kaggle. Dataset ini dikembangkan oleh GroupLens Research dan berisi data historis berupa penilaian (rating) film oleh pengguna pada layanan rekomendasi film MovieLens.
Secara keseluruhan, dataset ini memuat **100.836 data rating** yang diberikan oleh **610 pengguna** terhadap **9.742 film**. Setiap pengguna dalam dataset telah memberikan penilaian untuk sedikitnya 20 film. Periode data yang tercakup berlangsung dari **29 Maret 1996 hingga 24 September 2018**.
Dataset ini terdiri dari empat file utama, namun hanya dua file yang digunakan dalam proyek ini, yaitu `movies.csv` dan `ratings.csv`. Kedua file tersebut sudah bersih dari *missing values* dan duplikasi, sehingga dapat langsung digunakan untuk keperluan eksplorasi dan pembangunan model.

### Deskripsi File

#### `movies.csv`
File ini berisi metadata dari film yang tersedia.

- **Jumlah baris:** 9.742  
- **Jumlah kolom:** 3

**Variabel-variabel dalam `movies.csv`:**
- `movieId` : ID unik untuk setiap film, digunakan sebagai penghubung ke data rating.
- `title` : Judul lengkap film, biasanya mencakup tahun rilis di dalam tanda kurung.
- `genres` : Kategori genre dari film, ditulis dalam format string dan dipisahkan dengan simbol `|` (pipe).

#### `ratings.csv`
File ini menyimpan data interaksi pengguna dalam bentuk penilaian terhadap film.

- **Jumlah baris:** 100.836  
- **Jumlah kolom:** 4

**Variabel-variabel dalam `ratings.csv`:**
- `userId` : ID unik dari pengguna yang memberi penilaian.
- `movieId` : ID film yang dinilai oleh pengguna, berfungsi sebagai *foreign key* untuk menghubungkan dengan `movies.csv`.
- `rating` : Nilai penilaian yang diberikan oleh pengguna dalam skala 0.5 hingga 5.0 (dengan interval 0.5).
- `timestamp` : Waktu penilaian diberikan dalam format UNIX timestamp.

### Sumber Dataset
Dataset dapat diakses dan diunduh melalui tautan berikut:  
(https://www.kaggle.com/datasets/shubhammehta21/movie-lens-small-latest-dataset)


### Visualisasi Distribusi Rating Film
![image](https://github.com/user-attachments/assets/fb02cf04-309d-4478-bcb0-8c4c905228b2)

Visualisasi Distribusi Rating Film menampilkan frekuensi rating film yang diberikan. Terlihat bahwa rating 4.0 adalah yang paling sering diberikan, dengan jumlah lebih dari 25.000, menunjukkan bahwa banyak film menerima rating tinggi. Rating 3.0 juga memiliki frekuensi tinggi, sekitar 20.000. Rating yang lebih rendah (0.5 hingga 2.5) memiliki frekuensi yang jauh lebih sedikit dibandingkan rating yang lebih tinggi, mengindikasikan bahwa film-film cenderung mendapatkan rating yang cukup baik.

### Visualisasi Genre Film Terbanyak
![image](https://github.com/user-attachments/assets/1893dab5-63e1-4071-a6a6-5bff853c4e0e)

Visualisasi Genre Film Terbanyak menunjukkan 10 genre film dengan jumlah film terbanyak. Genre "Drama" mendominasi dengan lebih dari 1000 film, diikuti oleh "Comedy" dengan hampir 1000 film. Kombinasi genre seperti "Comedy|Drama" dan "Comedy|Romance" juga populer, menunjukkan preferensi penonton terhadap genre campuran. Genre-genre di bagian bawah daftar seperti "Horror" dan "Horror|Thriller" memiliki jumlah film yang relatif lebih sedikit.

### Visualisasi Genre dengan Jumlah Rating Terbanyak
![image](https://github.com/user-attachments/assets/2ab7b497-7d27-4e51-aa89-644ac1fd678e)

Visualisasi Genre dengan Jumlah Rating Terbanyak memperlihatkan 10 genre film yang menerima jumlah rating terbanyak. Sama seperti jumlah film terbanyak, genre "Comedy" dan "Drama" juga menerima jumlah rating terbanyak, dengan "Comedy" melebihi 7000 rating dan "Drama" mendekati 6500 rating. Hal ini menunjukkan bahwa genre-genre populer ini tidak hanya memiliki banyak film, tetapi juga secara aktif dinilai oleh penonton. Kombinasi genre seperti "Comedy|Romance" dan "Comedy|Drama|Romance" juga berada di peringkat atas, menunjukkan bahwa film-film dari genre ini sering mendapatkan rating.

## Data Preparation

Proses data preparation sangat penting untuk memastikan bahwa data dalam kondisi yang sesuai untuk digunakan dalam pemodelan sistem rekomendasi. Tahapan-tahapan berikut diterapkan secara berurutan untuk dua pendekatan utama: **Content-Based Filtering (CBF)** dan **Collaborative Filtering (CF)**.



###  a. Content-Based Filtering (CBF)

#### 1. Menggabungkan Data Rating dan Data Film

**Tujuan:** Menggabungkan dataset `ratings` dan `movies` berdasarkan `movieId` agar setiap entri rating dapat dikaitkan langsung dengan informasi film.

```python
data_merged = pd.merge(ratings, movies, on='movieId')
```

**Alasan:** Langkah ini penting agar setiap interaksi pengguna (rating) dapat dianalisis berdasarkan atribut film (judul dan genre), yang akan digunakan dalam proses ekstraksi fitur konten.



#### 2. Membersihkan Kolom Genre

**Tujuan:** Memformat kolom `genres` agar lebih mudah digunakan dalam ekstraksi fitur teks.

```python
data_merged['genres_clean'] = data_merged['genres'].str.replace('|', ' ')
```

**Alasan:** Pemisah `|` diganti menjadi spasi agar genre bisa diperlakukan seperti teks biasa saat diolah dengan teknik **TF-IDF**, yang memerlukan input berbentuk dokumen teks.



#### 3. Mengambil Data Film Unik

**Tujuan:** Menghapus duplikasi data berdasarkan `movieId`, `title`, dan `genres_clean`.

```python
movies = data_merged[['movieId', 'title', 'genres_clean']].drop_duplicates().reset_index(drop=True)
```

**Alasan:** Menghindari duplikasi yang dapat menyebabkan bias dalam perhitungan kemiripan antar film.



#### 4. Representasi Fitur Film dengan TF-IDF

**Tujuan:** Mengubah teks genre menjadi representasi numerik menggunakan TF-IDF.

```python
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres_clean'])
```

**Alasan:** TF-IDF membantu model mengenali genre yang unik dan informatif untuk masing-masing film, serta mengabaikan kata umum yang tidak relevan (stop words).



###  b. Collaborative Filtering (CF)

#### 1. Pemetaan ID Pengguna dan Film ke Indeks Numerik

**Tujuan:** Mengonversi `userId` dan `movieId` ke indeks integer untuk digunakan dalam model embedding neural network.

```python
unique_user_ids = data_merged['userId'].unique().tolist()
unique_movie_ids = data_merged['movieId'].unique().tolist()
user_id_to_index = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(unique_movie_ids)}

data_merged['user_index'] = data_merged['userId'].map(user_id_to_index)
data_merged['movie_index'] = data_merged['movieId'].map(movie_id_to_index)
```

**Alasan:** Model deep learning seperti RecommenderNet tidak dapat memproses ID langsung. Konversi ke indeks numerik memungkinkan penggunaan embedding layer secara efisien.



#### 2. Normalisasi Skor Rating (Min-Max Scaling)

**Tujuan:** Menstandarkan nilai rating dalam rentang 0–1 agar stabil untuk pelatihan model.

```python
rating_min = data_merged['rating'].min()
rating_max = data_merged['rating'].max()
data_merged['rating_scaled'] = data_merged['rating'].apply(lambda x: (x - rating_min) / (rating_max - rating_min))
```

**Alasan:** Nilai rating asli bervariasi. Normalisasi memastikan skala target homogen, sehingga pelatihan neural network lebih cepat konvergen.



#### 3. Mengacak dan Membagi Data Latih dan Validasi

**Tujuan:** Menghindari bias urutan dan menyiapkan data untuk pelatihan dan evaluasi model.

```python
data_merged = data_merged.sample(frac=1, random_state=42).reset_index(drop=True)
split_point = int(0.9 * len(data_merged))
train_data = data_merged.iloc[:split_point]
val_data = data_merged.iloc[split_point:]
```

**Alasan:** Data harus diacak untuk memastikan distribusi acak antara pelatihan dan validasi. Pembagian 90/10 umum digunakan untuk menjaga cukup data pada kedua subset.



#### 4. Menyiapkan Input dan Target

**Tujuan:** Memisahkan input dan target untuk pelatihan model.

```python
x_train_input = train_data[['user_index', 'movie_index']].values
y_train_target = train_data['rating_scaled'].values
x_val_input = val_data[['user_index', 'movie_index']].values
y_val_target = val_data['rating_scaled'].values
```

**Alasan:** Format ini disesuaikan dengan input model neural network, di mana input adalah pasangan (user, movie), dan target adalah skor rating terprediksi.

###

Proses data preparation dilakukan secara sistematis untuk memastikan kedua pendekatan (CBF dan CF) memiliki input data yang bersih, relevan, dan siap untuk proses pelatihan. Setiap langkah dirancang untuk meningkatkan akurasi dan stabilitas model rekomendasi.


## Modeling

Tahapan ini membahas proses pembangunan dua jenis model sistem rekomendasi: **Content-Based Filtering (CBF)** dan **Collaborative Filtering (CF)**. Output utama dari masing-masing pendekatan adalah rekomendasi Top-N untuk pengguna atau berdasarkan konten film tertentu.


###  a. Content-Based Filtering (CBF)

#### 1. Menghitung Skor Kemiripan Antar Film

Pada tahap ini, representasi fitur genre yang sebelumnya telah diproses dengan TF-IDF digunakan untuk menghitung **cosine similarity** antar film. Cosine similarity mengukur seberapa mirip dua film berdasarkan arah (sudut) dari vektor fitur mereka.

```python
# Menghitung skor kemiripan antar film berdasarkan vektor TF-IDF genre menggunakan cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
```

>  **Output:** Matriks kemiripan antar semua film yang digunakan untuk memberikan rekomendasi film yang mirip satu sama lain.



#### 2. Fungsi Rekomendasi Berdasarkan Konten

Fungsi `recommend_movies_cbf` dibuat untuk menerima judul film sebagai input, kemudian mengembalikan daftar Top-N film serupa berdasarkan kemiripan genre.

```python
# Membuat Series yang memetakan judul film ke indeks baris pada DataFrame 'movies'
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Fungsi untuk merekomendasikan film berdasarkan kemiripan konten (genre)
def recommend_movies_cbf(title, top_n=10):
    # Mengecek apakah judul film tersedia dalam data
    if title not in indices:
        return f"⚠️ Film '{title}' tidak ditemukan."

    # Mengambil indeks film berdasarkan judul
    idx = indices[title]

    # Mengambil skor kemiripan antara film tersebut dan semua film lainnya
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Mengurutkan berdasarkan skor kemiripan dari tertinggi ke terendah
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Mengambil indeks dari film dengan kemiripan tertinggi (lewati film itu sendiri)
    sim_scores = sim_scores[1:top_n + 1]
    movie_indices = [i[0] for i in sim_scores]

    # Mengembalikan DataFrame berisi film rekomendasi beserta genre-nya
    return movies.iloc[movie_indices][['title', 'genres_clean']]
```

> ✅ **Kelebihan CBF:** Tidak memerlukan data interaksi pengguna lain. Dapat memberikan rekomendasi bahkan untuk pengguna baru.
> ❌ **Kekurangan CBF:** Terbatas pada informasi konten; tidak menangkap pola preferensi pengguna.



###  b. Collaborative Filtering (CF)

#### 1. Membangun Model RecommenderNet Berbasis Embedding

Model dibangun dengan menggunakan pendekatan deep learning berbasis **neural collaborative filtering**. Model belajar dari pola interaksi pengguna-film menggunakan teknik embedding.

```python
# Ukuran dimensi embedding
EMBEDDING_SIZE = 50

# Bangun arsitektur Collaborative Filtering berbasis Embedding
class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super().__init__(**kwargs)

        # Embedding dan bias untuk user
        self.user_embedding = layers.Embedding(
            input_dim=num_users,
            output_dim=embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.user_bias = layers.Embedding(input_dim=num_users, output_dim=1)

        # Embedding dan bias untuk movie
        self.movie_embedding = layers.Embedding(
            input_dim=num_movies,
            output_dim=embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.movie_bias = layers.Embedding(input_dim=num_movies, output_dim=1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])

        dot_product = tf.reduce_sum(user_vector * movie_vector, axis=1, keepdims=True)
        x = dot_product + user_bias + movie_bias

        # Output diklip ke dalam range [0, 1]
        return tf.squeeze(tf.clip_by_value(x, 0.0, 1.0), axis=1)
```

>  **Tujuan:** Model belajar representasi vektor dari user dan movie, kemudian memprediksi preferensi pengguna melalui dot product embedding dan bias.



#### 2. Melatih Model Collaborative Filtering

Model dilatih menggunakan data interaksi pengguna dengan skema validasi. EarlyStopping digunakan untuk menghindari overfitting dan menyimpan bobot terbaik.

```python
# Inisialisasi model
model = RecommenderNet(total_users, total_movies, EMBEDDING_SIZE)
model.compile(
    loss='mse',
    optimizer=keras.optimizers.Adam(learning_rate=0.001)
)

# Callback EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Training model
history = model.fit(
    x=x_train_input,
    y=y_train_target,
    validation_data=(x_val_input, y_val_target),
    batch_size=64,
    epochs=15,
    callbacks=[early_stopping],
    verbose=1
)
```

📚 **Training Log (cuplikan):**
```
Epoch 1/15
1418/1418 - 16s - loss: 0.1741 - val_loss: 0.0539
Epoch 2/15
1418/1418 - 19s - loss: 0.0383 - val_loss: 0.0442
Epoch 3/15
1418/1418 - 11s - loss: 0.0257 - val_loss: 0.0443
Epoch 4/15
1418/1418 - 12s - loss: 0.0182 - val_loss: 0.0453
Epoch 5/15
1418/1418 - 21s - loss: 0.0140 - val_loss: 0.0467
✅ Training selesai!
```

> ✅ **Kelebihan CF:** Dapat menangkap pola preferensi kompleks dan memberikan rekomendasi personal. Tidak terbatas pada genre.
> ❌ **Kekurangan CF:** Tidak bekerja baik untuk user atau item baru (cold start), serta membutuhkan cukup banyak data interaksi.


###  Output: Top-N Recommendation

- **CBF:** Pengguna dapat memilih film favorit, lalu sistem akan memberikan **Top-N film serupa** berdasarkan genre.
- **CF:** Sistem dapat memberikan **Top-N film rekomendasi personal** untuk pengguna berdasarkan riwayat interaksi.


### ✅ Ringkasan

| Pendekatan | Kelebihan | Kekurangan |
|------------|-----------|------------|
| Content-Based Filtering | Tidak butuh data pengguna lain, bisa rekomendasi untuk user baru | Rekomendasi terbatas pada konten yang serupa |
| Collaborative Filtering | Rekomendasi lebih personal, mampu menangkap pola | Rentan terhadap cold start, butuh data besar |

Pendekatan gabungan atau **Hybrid Filtering** dapat digunakan untuk menggabungkan kekuatan keduanya dalam sistem rekomendasi yang lebih adaptif dan robust.



## Evaluation

### Evaluasi Kualitas Rekomendasi Content-Based Filtering (CBF)

**Hasil Evaluasi:**

- **Precision@5** (rata-rata dari 50 pengguna): **0.1600**  
- **MAP@5** (rata-rata dari 50 pengguna): **0.0991**

**Metrik yang Digunakan:**

- **Precision@k**: Proporsi item relevan di antara top-𝑘 hasil rekomendasi.
- **Mean Average Precision (MAP)@k**: Rata-rata dari precision@k, mempertimbangkan posisi item relevan dalam urutan.

**Formula:**

$$
\text{Precision@k} = \frac{\text{Jumlah item relevan dalam top-k}}{k}
$$

$$
\text{Average Precision@k} = \frac{1}{\min(|\text{relevant}|, k)} \sum_{i=1}^{k} P(i) \times \text{rel}(i)
$$


![download](https://github.com/user-attachments/assets/6b9498e0-329d-4723-ba97-bd16a0eed0ad)

Grafik ini menunjukkan hasil evaluasi sistem Content-Based Filtering (CBF) berdasarkan metrik **Precision@K** dan **MAP@K** (Mean Average Precision).

- **Precision@K** mengukur seberapa banyak item yang direkomendasikan dalam Top-K yang relevan.
- **MAP@K** menghitung rata-rata precision untuk semua K dan semua user.

#### Insight:
- Nilai Precision@K dan MAP@K menurun seiring bertambahnya nilai K.
- Hal ini umum terjadi karena semakin banyak item yang direkomendasikan, kemungkinan item yang tidak relevan meningkat.


  
### Evaluasi Model Collaborative Filtering (CF)

**Hasil Evaluasi Model:**

- **RMSE**: 0.9270  
- **MAE** : 0.7108  

**Metrik yang Digunakan:**

- **Root Mean Squared Error (RMSE)**: Mengukur seberapa jauh prediksi model dari nilai rating sebenarnya dalam skala kuadrat.
- **Mean Absolute Error (MAE)**: Rata-rata dari selisih absolut antara rating prediksi dan rating aktual.

**Formula:**

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

Evaluasi ini menunjukkan bahwa model CF memiliki kesalahan prediksi yang cukup rendah, mengindikasikan performa yang baik dalam memprediksi rating pengguna.

![image](https://github.com/user-attachments/assets/03592cc2-6b7e-440a-b16f-1c7f9f56e0dc)

Grafik ini menggambarkan perubahan nilai loss selama proses pelatihan model (misalnya neural network).

- **Training Loss** menunjukkan error model terhadap data pelatihan.
- **Validation Loss** menunjukkan error terhadap data validasi (untuk mengecek overfitting).

#### Insight:
- Training loss terus menurun, yang menunjukkan model belajar dari data pelatihan.
- Validation loss sempat menurun lalu sedikit meningkat, mengindikasikan **awal mula overfitting** setelah epoch ke-2.



### Inference & User Input

Fungsi berikut memberikan dua opsi rekomendasi film berdasarkan input pengguna:

- **Content-Based Filtering (CBF)**  
  - Rekomendasi berdasarkan kemiripan genre film dengan cosine similarity.
  - Input: Judul film favorit.
  
- **Collaborative Filtering (CF)**  
  - Rekomendasi berdasarkan prediksi rating pengguna dengan model neural network.
  - Input: ID pengguna.

Pengguna dapat memilih metode sesuai dengan preferensi atau ketersediaan data.
![image](https://github.com/user-attachments/assets/e55c3e14-79b5-4e49-93a6-45d49e5e922d)

#### Perbandingan Top-10 Film Rekomendasi: CBF vs CF

Grafik ini menampilkan hasil rekomendasi sistem berdasarkan dua pendekatan:
- **CBF (Content-Based Filtering)**: Rekomendasi berdasarkan kemiripan konten (fitur film).
- **CF (Collaborative Filtering)**: Rekomendasi berdasarkan kesamaan preferensi user.

##### CBF - Mirip dengan "Toy Story (1995)":
- Semua film yang direkomendasikan memiliki kesamaan genre, gaya animasi, atau segmentasi usia.

##### CF - Untuk User ID 1:
- Film yang direkomendasikan berasal dari berbagai genre populer berdasarkan kesamaan rating pengguna lain.

##### Insight:
- CBF cenderung merekomendasikan film dengan konten serupa.
- CF lebih personal dan berdasarkan perilaku user lain, memberikan hasil rekomendasi yang lebih beragam.

  
## Conclusion
Sistem rekomendasi yang dibangun berhasil memberikan hasil yang relevan dan personalisasi menggunakan dua pendekatan utama, yaitu Content-Based Filtering (CBF) dan Collaborative Filtering (CF).

### Content-Based Filtering (CBF)
- Cocok untuk pengguna yang sudah memiliki riwayat interaksi atau film favorit.
- Memberikan rekomendasi berdasarkan kemiripan konten film yang pernah disukai, seperti genre atau atribut film lainnya.
- Kurang efektif untuk pengguna baru tanpa riwayat karena membutuhkan data preferensi pengguna.

### Collaborative Filtering (CF)
- Efektif saat tersedia data interaksi dan rating dari banyak pengguna.
- Menghasilkan rekomendasi berdasarkan pola preferensi dan perilaku pengguna lain yang mirip.
- Untuk pengguna baru (cold-start), CF memiliki keterbatasan kecuali didukung dengan teknik seperti popularitas atau model hybrid.

### Evaluasi
- Precision@5 dan MAP@5 menunjukkan hasil yang cukup baik untuk CBF.
- Model CF memberikan performa yang memuaskan dengan RMSE dan MAE yang wajar.

### Potensi Pengembangan di Masa Depan
- **Hybrid Filtering:** Menggabungkan keunggulan CBF dan CF untuk hasil rekomendasi yang lebih akurat dan stabil.
- **Deep Learning Berbasis Sequence:** Memanfaatkan model RNN atau Transformer untuk menangkap pola interaksi pengguna secara lebih kompleks dan dinamis.
- **Context-Aware Recommender:** Menambahkan konteks seperti waktu, perangkat, dan lokasi pengguna untuk meningkatkan personalisasi dan relevansi rekomendasi.
Dengan implementasi dan pengembangan berkelanjutan, sistem rekomendasi film ini dapat terus meningkat dan memberikan pengalaman pengguna yang lebih memuaskan.

## Diagram Sistem
Berikut adalah diagram visualisasi sistem rekomendasi film yang mendukung pemahaman analisis dan arsitektur:
###Data Flow Diagram (DFD)
<img width="1423" height="330" alt="graphviz (1)" src="https://github.com/user-attachments/assets/4cbad106-652f-4901-a94e-8544f25292ac" />
###Entity Relationship Diagram (ERD)
<img width="781" height="373" alt="graphviz (2)" src="https://github.com/user-attachments/assets/09e1d476-40e2-41fd-b826-e297e1ff3bb1" />
###UML Use Case Diagram
<img width="1288" height="418" alt="graphviz (3)" src="https://github.com/user-attachments/assets/8245f691-62db-4bb9-a249-ccb89c1a6142" />
###UML Sequence Diagram
<img width="1345" height="270" alt="graphviz (4)" src="https://github.com/user-attachments/assets/02d9a4ea-3d12-43b3-9090-1ef76b9232b8" />
###Diagram Arsitektur Proses Hybrid Filtering
<img width="990" height="997" alt="graphviz" src="https://github.com/user-attachments/assets/6883de29-64e8-49fd-b95f-4301a001be4b" />

