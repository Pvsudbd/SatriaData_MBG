### Kelompok Erid
## Case 1
## Model training : XLM-RoBERTa dengan pendekatan Cleanlab, TFidf sebagai probabilitas, dan resample untuk membantu data kelas minoritas

# Kebutuhan pustaka
torch>=2.0.0
transformers>=4.30.0
cleanlab>=2.0.0
scikit-learn>=1.0.0
pandas>=1.5.0
numpy>=1.20.0
accelerate>=0.20.0
openpyxl>=3.0.0

# Kebutuhan pustaka dari library dan kegunaan
### 🛠️ Kebutuhan Pustaka (Dependencies)

Seluruh pustaka yang digunakan dalam proyek ini wajib diinstal terlebih dahulu. Berikut adalah daftar *library* beserta komponen spesifik yang digunakan di dalam *source code*:

#### 1. Core Frameworks & Deep Learning
* **`os`**: Untuk ngegganti running model menggunakan GPU (Tergantung laptop) (`os.environ["CUDA_VISIBLE_DEVICES"] = "0"`).
* **`torch` (PyTorch)**: Framework utama untuk komputasi *Deep Learning* berbasis tensor, manajemen memori GPU CUDA, serta konversi tipe data larik ke format tensor (`torch.tensor`, `torch.float`, `torch.long`).

#### 2. Hugging Face Transformers
* **`AutoTokenizer`**: Digunakan untuk memuat arsitektur pengurai teks (*SentencePiece Tokenizer*) bawaan dari model *pre-trained* `xlm-roberta-base`.
* **`AutoModelForSequenceClassification`**: Digunakan untuk mengunduh arsitektur model neural network XLM-RoBERTa yang sudah siap dikonfigurasi untuk tugas klasifikasi teks (*sequence classification*).
* **`TrainingArguments` & `Trainer`**: Komponen utama untuk mengatur jalannya alur pelatihan (*training loop*), *hyperparameters*, optimasi, dan proses evaluasi model secara otomatis.
* **`EarlyStoppingCallback`**: Mekanisme penyelamat untuk menghentikan pelatihan secara otomatis jika performa model pada data validasi sudah tidak mengalami peningkatan, demi mencegah terjadinya *overfitting*.

#### 3. Cleanlab (Data-Centric AI)
* **`find_label_issues`**: Algoritma inti dari pustaka Cleanlab untuk menganalisis ketidakselarasan antara label asli manusia dengan probabilitas prediksi mesin, guna menyaring dan mendeteksi label-label yang kotor/salah (*noisy labels*).

#### 4. Scikit-Learn (Machine Learning Toolkit)
* **`TfidfVectorizer`**: Digunakan untuk mengekstrak fitur teks mentah menjadi representasi angka berbasis statistik kata (TF-IDF) sebagai input untuk model pembantu.
* **`LogisticRegression`**: Model klasifikasi linear cepat yang bertugas menghasilkan matriks probabilitas tebakan awal (*predict_proba*) yang objektif untuk kebutuhan Cleanlab dan Soft Label.
* **`cross_val_predict`**: Menjalankan skema *5-Fold Cross Validation* saat memprediksi probabilitas data latih agar tebakan model pembantu bersifat jujur dan tidak bias (*out-of-sample*).
* **`LabelEncoder`**: Mengonversi teks kategori label (`Anggaran`, `Politik`, dll) menjadi indeks angka (`0` sampai `7`), serta memetakan balik angka tebakan menjadi teks saat tahap akhir (`inverse_transform`).
* **`train_test_split`**: Membagi dataset bersih menjadi porsi 80% data latihan dan 20% data validasi secara acak namun tetap proporsional (`stratify`).
* **`resample`**: Strategi *Random Oversampling* dengan pengembalian (`replace=True`) untuk memperbanyak jumlah sampel pada kelas minoritas yang kritis (*Ekonomi* dan *Distribusi*).
* **`compute_class_weight`**: Menghitung bobot penalti otomatis untuk tiap-tiap kelas berdasarkan kelangkaan datanya untuk disuapkan ke dalam fungsi kerugian (*Loss Function*) model.
* **`balanced_accuracy_score`**: **Metrik penilaian utama kompetisi** untuk menghitung nilai rata-rata dari *Recall* setiap kelas secara adil pada distribusi data yang tidak seimbang.
* **`classification_report`**: Menampilkan tabel evaluasi komprehensif yang memuat rincian nilai *Precision*, *Recall*, dan *F1-Score* untuk masing-masing dari 8 kelas.

#### 5. Data Manipulation & Utils
* **`pandas`**: Membaca file dataset tabular CSV dari panitia, memanipulasi struktur baris/kolom teks, serta mengekspor hasil akhir ke format Excel.
* **`numpy`**: Melakukan operasi matriks tingkat tinggi, seperti konversi label kaku menjadi matriks peluang (`np.eye`), penggabungan array vertikal (`np.vstack`), pencampuran rumus Soft Label (*Alpha Formula*), serta pengacakan baris data (`np.random.RandomState`).
* **`product` (dari `itertools`)**: Digunakan untuk menghasilkan kombinasi tak terbatas (*Cartesian Product*) saat melakukan eksperimen pencarian nilai batas optimal (*Grid Search Threshold Tuning*).

> Bisa di download pada requirement.txt


# Bagaimana cara memulai training dan bagaimana training tersebut bekerja
> Jalankan langkah sesuai urutan nomer!.
1. Buka `PreprocessingText` untuk melakukan cleaning noise pada data berlabel maupun data yang ingin diberi label 
2. Buka `XLM_fixed_v3.ipynb` 
3. Install & Import Dependencies seperti torch, transformers, cleanlab, scikit-learn, pandas
4. Load data dari `dataset2/train_clean_v2.csv` 
5. Jalankan proses encoder agar dapat membuat id untuk setiap label. 
6. Selanjutnya, program mendeteksi kesalahan input label (*human error/annotation noise*) pada data latih menggunakan pendekatan matematika murni melalui pustaka `Cleanlab` dengan alur sebagai berikut:
    - **Vektorisasi Teks (`TfidfVectorizer`):** Teks tweet mentah diubah menjadi angka matriks bobot kata berdasarkan tingkat kepentingan kata tersebut (maksimal 10,000 fitur kata terpenting).
    - **Prediksi Probabilitas Awal (`LogisticRegression`):** Model statistik cepat (*Logistic Regression*) dilatih menggunakan teknik *5-Fold Cross Validation* (`cv=5`). Tujuannya untuk mendapatkan tebakan peluang (*probabilitas*) yang jujur dan objektif untuk setiap baris kalimat di dataset (`pred_probs_tfidf`).
   - **Analisis Ketiadakyakinan (`find_label_issues`):** Fungsi dari `Cleanlab` akan mengawinkan antara *label asli dari manusia* dengan *probabilitas tebakan mesin*. Jika mesin sangat yakin sebuah tweet adalah kelas "Ekonomi" (probabilitas tinggi) tapi di label aslinya tertulis "Politik", Cleanlab akan menandai baris tersebut sebagai kejanggalan (*Label Issues*).
   - **Flagging Data (`is_noisy`):** Indeks data yang terdeteksi salah label akan ditandai dengan status `is_noisy = True`. Selisih data yang tidak bermasalah akan diisolasi sebagai **Data Bersih** yang siap digunakan untuk proses *Oversampling* dan pelatihan model utama XLM-RoBERTa tanpa takut terganggu oleh data yang sesat.

7. Pada tahap ini, program melakukan manipulasi tingkat lanjut pada data agar model XLM-RoBERTa bisa belajar secara adil dan kebal terhadap kesalahan label:
   - **Pembuatan Soft Label (`ALPHA = 0.7`):** Program mengubah label kaku manusia (*One-Hot Encoding*) menjadi label yang lebih fleksibel (*Soft Label*). Dengan bobot 70% percaya pada label manusia dan 30% percaya pada probabilitas TF-IDF, formula ini bertujuan untuk melunakkan label yang ambigu sehingga model tidak dipaksa menghafal data yang berpotensi salah.
   - **Isolasi Data Bersih (`clean_mask`):** Berdasarkan hasil deteksi Cleanlab di tahap sebelumnya, semua baris teks yang diberi label `is_noisy = True` akan dibuang secara permanen dari data *training*. Hanya data yang 100% terkonfirmasi bersih (`data_clean`) yang lolos ke tahap berikutnya.
   - **Oversampling Kelas Minoritas (`resample`):** Untuk mengatasi ketimpangan jumlah data, kelas *Ekonomi* diduplikasi secara acak hingga mencapai 450 baris dan kelas *Distribusi* hingga 650 baris menggunakan fungsi `resample(replace=True)`. Langkah ini krusial agar model memiliki kesempatan belajar yang cukup pada kelas-kelas yang langka.
   - **Penggabungan dan Pengacakan Data (*Shuffle*):** Setelah kelas minoritas diperbanyak, seluruh kelompok data (baik yang di-oversample maupun kelas lain yang tidak disentuh) digabungkan kembali menjadi satu kesatuan. Data tersebut kemudian diacak posisinya secara acak berdasarkan nilai `RANDOM_SEED = 42` agar urutan barisnya tidak monoton saat dibaca oleh model sewaktu pelatihan berjalan.

8. Pada tahap **Train/Val split**, dataset yang sudah bersih dan seimbang (`data` dan `soft_labels`) dibagi menjadi dua porsi terpisah menggunakan fungsi `train_test_split` dengan aturan yang ketat:
   - **Proporsi Pembagian (`test_size=0.2`):** Dataset dibagi dengan rasio 80% untuk data `Train` (dipakai model untuk belajar mencari pola kata) dan 20% untuk data `Val` / Validasi (dipakai sebagai lembar ujian untuk mengetes kepintaran model setelah belajar).
   - **Pembagian yang Adil (`stratify=data['label_id']`):** Parameter ini wajib digunakan agar distribusi atau persentase tiap kelas di data `Train` dan data `Val` benar-benar sama dan merata. Jika di data total kelas Politik ada 15%, maka di data ujian dan data belajar jumlahnya akan diset pas 15% juga.
   - **Konsistensi Data (`random_state=RANDOM_SEED`):** Mengunci angka acak menggunakan nilai 42 agar jika kode ini dijalankan ulang oleh panitia, hasil pembagian baris datanya akan selalu sama persis (*reproducible*).
   - **Merapikan Baris (`reset_index`):** Menghapus indeks baris yang berantakan akibat proses pengacakan sebelumnya dan mengurutkannya kembali dari angka 0, sehingga struktur *Dataframe* menjadi rapi dan siap dimasukkan ke dalam tokenizers XLM-RoBERTa.

9. Pada tahap **Tokenisasi**, teks tweet yang masih berupa string/kata-kata manusia diubah menjadi format angka matriks (*input_ids* dan *attention_mask*) yang bisa dipahami dan diproses oleh arsitektur model neural network XLM-RoBERTa:
   - **Aturan Pembatasan (`max_length=128`):** Batas maksimal panjang token untuk setiap tweet dikunci di angka 128 token. Angka ini dipilih karena sudah sangat ideal untuk menampung panjang rata-rata sebuah tweet tanpa membuang memori GPU secara berlebihan.
   - **Penyelarasan Panjang Teks (`padding=True` & `truncation=True`):** - *Padding:* Jika ada tweet yang pendek (misal cuma 10 kata), komputer akan otomatis menambah angka `0` di belakangnya sampai panjangnya pas 128.
     - *Truncation:* Jika ada tweet yang terlalu panjang (melebihi 128 token), kata-kata di ujung belakangnya akan dipotong secara otomatis.
   - **Eksekusi Encodings:** Fungsi `tokenize` dijalankan pada data latih (`train_df`) dan data validasi (`valid_df`) untuk menghasilkan variabel `train_encodings` dan `valid_encodings` yang sudah matang dan siap disuapkan ke dalam model saat proses *fine-tuning*.

10. Pada tahap **Dataset Class**, program membungkus token teks (*encodings*), label asli manusia (*hard labels*), dan probabilitas campuran (*soft labels*) ke dalam sebuah struktur data kustom bernama `SoftLabelDataset` berbasis kelas `torch.utils.data.Dataset`:
    -**Function penampung dua label**(`__init__`):** Kelas ini didesain secara kustom agar fleksibel menampung dua format penilaian sekaligus, yaitu `labels` (berbentuk angka integer/kaku untuk evaluasi biasa) dan `soft_labels` (berbentuk *array desimal/float* hasil rumus campuran TF-IDF untuk proses *training*).
    - **Ekstraksi Data Per Baris (`__getitem__`):** Fungsi ini otomatis mengubah data teks yang sudah ditokenisasi menjadi bentuk *PyTorch Tensor* (`torch.tensor`). Setiap kali model meminta data, fungsi ini akan mengirimkan paket lengkap berisi `input_ids`, `attention_mask`, `labels` (tipe data *long*), serta `soft_labels` (tipe data *float*, jika tersedia).
    - **Penghitungan Total Data (`__len__`):** Berfungsi untuk memberi tahu model secara tepat berapa jumlah total sampel data yang ada di dalam antrean untuk diproses.
    - **Pembedaan Data Latih dan Validasi:** - `train_dataset` dibuat dengan menyertakan `soft_labels=soft_train` agar model bisa memanfaatkan mitigasi *noisy label* saat belajar.
      - `valid_dataset` dibuat **tanpa** menyertakan *soft label* karena data validasi murni bertindak sebagai lembar ujian standar untuk menguji akurasi prediksi aslinya.
    
11. Tahap **Class Weight & boost label minoritas**
Setelah data di-oversample, model tetap diberi sinyal tambahan melalui class weight agar lebih sensitif terhadap kelas minoritas. Class weight dihitung otomatis menggunakan `compute_class_weight('balanced')` dimana kelas yang lebih sedikit datanya, otomatis mendapat bobot lebih besar.

    - Contoh hasil class weight sebelum boost:
    Anggaran        : 0.821
    Distribusi      : 0.654
    Ekonomi         : 1.243   <- fokus ke ini
    Kualitas Pangan : 0.512
    -  Karena Ekonomi masih sering missed meski sudah di-oversample maka saya melakukan boost bobot Ekonomi secara manual 2x:
    Ekonomi setelah boost : 2.486
     alasan saya memboost nilai ini adalah karena meskipun class weight sudah dihitungotomatis secara balanced dan data Ekonomi sudah di-oversample, model masih cenderung mengabaikan kelas ini akibat jumlah sampelnya yang sangat sedikit (145 dari 5000 = 2.9%). Boost 2x memaksa model memberi perhatian ekstra pada Ekonomi tanpa terlalu mengorbankan akurasi kelas lain.

> Note : Semakin sedikit data suatu kelas, semakin besar bobot yang diberikan. Tanpa class weight, model cenderung mengabaikan Ekonomi karena model beranggapan "lebih aman" selalu prediksi kelas mayoritas karena akurasinya tetap tinggi Meskipun Ekonomi tidak pernah diprediksi sama sekali.Dengan class weight, model "dipaksa" memperhatikan semua kelas secara proporsional.

12. Pada tahap **Training**, model XLM-RoBERTa dilatih menggunakan `NoisyAwareTrainer` dengan konfigurasi sebagai berikut:

    - `learning_rate=2e-5` → learning rate kecil agar model fine-tuning secara stabil 
      tanpa merusak bobot pretrained
    - `warmup_ratio=0.1` → 10% awal training digunakan untuk menaikkan learning rate 
      secara bertahap sebelum mulai turun mengikuti cosine scheduler
    - `lr_scheduler_type='cosine'` → learning rate turun mengikuti kurva cosine, 
      lebih smooth dibanding linear
    - `num_train_epochs=10` → maksimal 10 epoch, namun training bisa berhenti lebih 
      awal karena early stopping
    - `weight_decay=0.05` → regularisasi untuk mencegah overfitting
    - `early_stopping_patience=2` → training otomatis berhenti jika balanced accuracy 
      tidak meningkat selama 2 epoch berturut-turut
    - `load_best_model_at_end=True` → setelah training selesai, bobot model yang 
      dipakai adalah yang menghasilkan balanced accuracy tertinggi di validation set, 
      bukan epoch terakhir

> Metrik utama yang dipantau adalah `balanced_accuracy`, karena data imbalanced, sehingga akurasi biasa tidak cukup representatif. Balanced accuracy menghitung rata-rata recall per kelas, sehingga kelas kecil seperti Ekonomi tetap berkontribusi sama besarnya dengan kelas mayoritas.

13. Pada tahap **Evaluasi**, program melakukan simulasi penilaian performa model menggunakan metrik resmi yang ditetapkan oleh panitia lomba, yaitu **Balanced Accuracy**:
    - **Ekstraksi Logits dan Softmax:** Hasil prediksi mentah (*logits*) dari model dikonversi menjadi nilai probabilitas (persentase 0-100%) menggunakan fungsi `torch.softmax`.
    - **Penentuan Kelas Terkuat (`np.argmax`):** Untuk evaluasi awal (*baseline*), sistem mengambil label dengan nilai probabilitas tertinggi sebagai hasil tebakan akhir.
    - **Visualisasi Laporan (`classification_report`):** Menampilkan tabel performa komprehensif berisi skor *Precision*, *Recall*, dan *F1-Score* per kelas untuk menganalisis kelas mana saja yang sudah kuat atau masih lemah sebelum masuk ke tahap *Threshold Tuning*.

14. Pada tahap **Threshold Tuning**, setiap kelas diberi threshold berbeda untuk mengoptimalkan balanced accuracy, khususnya kelas lemah.
Cara kerjanya: probabilitas prediksi tiap kelas dibagi dengan threshold-nya 
    
Model prediksi satu tweet:
Ekonomi : 0.15,  Distribusi : 0.20,  Politik : 0.65

Tanpa threshold (default 1.0):
adjusted = [0.15/1.0, 0.20/1.0, 0.65/1.0] = [0.15, 0.20, 0.65]
→ prediksi: Politik  ✗ (padahal harusnya Ekonomi)

Dengan threshold Ekonomi = 0.3:
adjusted = [0.15/0.3, 0.20/1.0, 0.65/1.0] = [0.50, 0.20, 0.65]
→ prediksi: Politik  ✗ (masih kalah)

Dengan threshold Ekonomi = 0.2:
adjusted = [0.15/0.2, 0.20/1.0, 0.65/1.0] = [0.75, 0.20, 0.65]
→ prediksi: Ekonomi  ✓

>  Threshold yang lebih kecil = kelas lebih mudah dipilih. Oleh karena itu Ekonomi diberi opsi threshold paling agresif [0.2, 0.3, 0.4, 0.5, 0.6], dibandingkan kelas lain yang  [0.6, 0.7, 0.8, 0.9, 1.0]. Semua kombinasi threshold dicoba via `itertools.product` dan dipilih kombinasi yang menghasilkan balanced accuracy tertinggi di validation set.

15. Setelah threshold terbaik ditemukan, prediksi final dijalankan ulang menggunakan `best_thresh` dan dievaluasi dengan `classification_report` yang menampilkan precision, recall, dan f1-score per kelas.
    - **Precision** → dari semua yang diprediksi kelas X, berapa yang benar
    - **Recall**    → dari semua data kelas X, berapa yang berhasil ditemukan model
    - **F1-score**  → rata-rata harmonis precision dan recall

16. Untuk OOF mungkin bisa di skip karena model yang kami pilih untuk dikumpulkan di lomba adalah XLM roBERTa

| Model | Metode | Balanced Accuracy |
|---|---|---|
| Complement Naive Bayes | TF-IDF + Naive Bayes | 58% |
| BERT | Fine-tuning `bert-base` | 64% |
| IndoBERTweet | Fine-tuning IndoBERTweet | 69% |
| **XLM-RoBERTa** (ours) | Fine-tuning + Soft Label + NoisyAwareTrainer + Threshold Tuning | **79-80%*** |

