## sistem klasifikasi emosional berdasarkan teks

# Deskripsi Proyek

Proyek ini merupakan sistem klasifikasi emosi berbasis Natural Language Processing (NLP) yang dibangun menggunakan dataset GoEmotions.csv , yang dimana data ini saya ambil langsung dari kaggel.

untuk sistem ini dapat menerima teks baru dari pengguna dan mengklasifikasikan emosi yang terkandung dalam teks tersebut ke dalam salah satu dari 28 kategori emosi yang terdapat pada label data.

Model ini dibangun menggunakan metode TF-IDF sebagai representasi fitur teks dan Logistic Regression sebagai algoritma klasifikasi.



# Dataset

Dataset yang digunakan adalah **GoEmotions Dataset**.

Karakteristik dataset:

- Jumlah data: 211.225 teks
- Jumlah label emosi: 28
- Bahasa:  data ini sudah menggunakan bahasa Inggris
- Tipe data: Multi-label 

Contoh label emosi:

- admiration
- amusement
- anger
- annoyance
- approval
- caring
- confusion
- curiosity
- desire
- disappointment
- disapproval
- disgust
- embarrassment
- excitement
- fear
- gratitude
- grief
- joy
- love
- nervousness
- optimism
- pride
- realization
- relief
- remorse
- sadness
- surprise
- neutral




# Tahapan NLP

1. Preprocessing Text

Tahapan preprocessing yang dilakukan:

- Case Folding (mengubah huruf menjadi bentuk huruf kecil semua)
- Menghapus URL
- Menghapus angka
- Menghapus tanda baca
- Menghapus spasi berlebih

2. Representasi Fitur

Metode yang digunakan:

- TF-IDF (Term Frequency – Inverse Document Frequency)

Jumlah fitur:

- 5000 fitur

3. Training Model

Algoritma yang digunakan:

- Logistic Regression

Pembagian data:

- Training Data: 80%
- Testing Data: 20%

4. Evaluasi

Metrik evaluasi yang digunakan:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Hasil evaluasi:

- Accuracy: dari hasil akurasi yang didapat yaitu sebesar 39.41%



# Cara Menjalankan Proyek

1. Clone Repository

git clone <repository-url>


2. Masuk ke Folder Proyek

cd emotion-classification


3. Install Dependency

pip install -r requirements.txt

4. Jalankan Aplikasi

streamlit run app.py

5. Buka Browser

http://localhost:8501  --> ketika masih di run didalam localhost


# Dokumentasi Antarmuka

Untuk mempermudah kami dalam membuat tampilan dari aplikasi ini dan juga memberikan suatu tampilan yang bisa dilihat pengguna, kami menggunakan sebuah framework yaitu Streamlit. Mengapa kami memilih menggunakan streamlit agar tidak perlu lagi melakukan REST API  dan juga tampilan yang kami buat juga masih sederhana banget sehingga tidak perlu memerlukan yang lebih detail.


1. Input

Pengguna memasukkan teks ke dalam kolom input.

Contoh:

I am very happy today.


2. Proses

Sistem akan:

1. Melakukan preprocessing teks.
2. Mengubah teks menjadi representasi TF-IDF.
3. Menggunakan model Logistic Regression untuk melakukan prediksi emosi.

3. Output

Sistem menampilkan:

- Label emosi hasil prediksi
- Emoji yang merepresentasikan emosi
- diagram yang menampilkan perbandingan dari setiap emosi

Contoh output:


JOY 😄 ( beserta diagram )



# Teknologi yang Digunakan

- Python
- Pandas
- Scikit-Learn
- TF-IDF
- Logistic Regression
- Streamlit
- Joblib
