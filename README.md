# Mendeteksi Churn, Menjaga Nilai Pelanggan

Analisis klasifikasi pelanggan telko menggunakan machine learning, disertai simulasi biaya–manfaat untuk mendukung keputusan retensi pelanggan.

## 1. Overview Project

Churn terjadi ketika pelanggan berhenti menggunakan layanan perusahaan. Mengetahui pelanggan yang berpotensi churn dapat membantu perusahaan menentukan prioritas kontak dan penawaran retensi. Namun, kualitas prediksi perlu dinilai bersama biaya kampanye dan kemampuan intervensi dalam mempertahankan pelanggan.

Project ini membandingkan enam algoritma supervised classification, memperbaiki model terpilih, dan menerjemahkan hasil prediksi menjadi estimasi manfaat bisnis. Analisis berfokus pada pemilihan model, generalisasi, penanganan ketimpangan kelas, serta kelayakan ekonomi kampanye.

### Dataset

- **Sumber:** file `data_telco_customer_churn.csv` yang disediakan untuk project ini.
- **Ukuran:** 4.930 observasi dengan 10 fitur prediktor dan satu target.
- **Target:** `Churn`, dengan `Yes = 1` dan `No = 0`.
- **Proporsi churn:** sekitar 26,7% dari seluruh observasi.
- **Pembagian data:** 80% data latih atau 3.944 observasi dan 20% data uji atau 986 observasi, menggunakan stratifikasi target dan `random_state=42`.

Seluruh fitur prediksi berasal dari dataset tersebut. Asumsi biaya, kontribusi pelanggan, efektivitas retensi, dan volume rollout dibuat khusus untuk simulasi bisnis; nilainya tidak tersedia di dataset.

## 2. Objective Project

### Tujuan Pemodelan

1. Memeriksa tipe data, missing value, distribusi numerik, dan hubungan antarfitur.
2. Membandingkan enam algoritma menggunakan validasi silang pada data latih.
3. Memilih model berdasarkan **F1-score**, dengan ROC AUC, precision, recall, dan accuracy sebagai metrik pendukung.
4. Mengevaluasi generalisasi dan ketimpangan kelas, kemudian menguji oversampling, feature engineering, hyperparameter tuning, dan threshold.
5. Membandingkan hasil model awal dan model perbaikan pada data uji, termasuk TP, TN, FP, FN, false-positive rate, dan specificity.

### Tujuan Cost and Benefit Analysis

Menghitung apakah kampanye berbasis model perbaikan menghasilkan manfaat bersih lebih besar daripada kampanye berbasis model awal setelah memperhitungkan biaya implementasi dan operasional.

| Kelompok | Komponen simulasi |
|---|---|
| Biaya langsung | Sewa komputasi, penyimpanan, integrasi CRM, pelatihan petugas, pemeliharaan, kontak dan penawaran retensi |
| Biaya tidak langsung | Waktu adaptasi di luar pelatihan dan penanganan gangguan akibat kontak salah sasaran |
| Manfaat langsung | Kontribusi pelanggan yang berhasil dipertahankan dan penghematan waktu seleksi manual |
| Manfaat tidak langsung | Kontribusi bersih pelanggan baru dari referral, sebagai asumsi opsional |

Setiap item operasional dihitung dari **jumlah kegiatan × tarif**. Contohnya, integrasi CRM menggunakan jumlah jam kerja, sedangkan pelatihan menggunakan jumlah peserta × durasi pelatihan.

Rumus utama:

```text
Pelanggan dihubungi = TP + FP
Ekspektasi pelanggan dipertahankan = TP × efektivitas pencegahan churn
Kontribusi retensi = ekspektasi pelanggan dipertahankan × kontribusi bulanan × tambahan bulan bertahan
Biaya kampanye = pelanggan dihubungi × biaya kontak dan penawaran

Total manfaat = manfaat langsung + manfaat tidak langsung
Total biaya = biaya langsung + biaya tidak langsung
Manfaat bersih = total manfaat − total biaya
Tambahan manfaat bersih = manfaat bersih model perbaikan − manfaat bersih model awal
ROI = manfaat bersih / total biaya
```

Biaya kontak false positive sudah termasuk dalam biaya kampanye. Peluang retensi yang hilang akibat false negative ditampilkan terpisah agar tidak dihitung dua kali dalam manfaat bersih.

## 3. Tools, Metode, dan Library

### Tools dan Library

Notebook menggunakan Python dan dapat dijalankan melalui Jupyter Notebook, Google Colab, atau VS Code dengan dukungan notebook.

| Library | Penggunaan |
|---|---|
| pandas, NumPy | Pengolahan data dan perhitungan numerik |
| Matplotlib, seaborn | Visualisasi distribusi, evaluasi model, dan simulasi bisnis |
| SciPy | Pemeriksaan distribusi dan uji Shapiro–Wilk |
| statsmodels | Q–Q plot dan Variance Inflation Factor atau VIF |
| scikit-learn | Preprocessing, pipeline, klasifikasi, validasi silang, tuning, dan evaluasi |
| imbalanced-learn | RandomOverSampler dan pipeline yang mendukung resampling |
| LightGBM | Model boosting tambahan, dapat dinonaktifkan melalui `USE_LIGHTGBM` |
| IPython, ipykernel | Tampilan tabel dan eksekusi notebook |
| nbformat, nbconvert | Pengelolaan format notebook dan ekspor laporan HTML |

Versi dependensi tersedia di [`requirements.txt`](requirements.txt).

### Algoritma yang Diuji

1. Logistic Regression
2. K-Nearest Neighbors atau KNN
3. Decision Tree
4. Random Forest
5. HistGradientBoosting
6. LightGBM

### Alur Analisis

1. **Pemeriksaan data:** tipe data, nilai kosong, duplikasi, dan konsistensi kategori.
2. **Distribusi numerik:** statistik deskriptif, histogram, Q–Q plot, skewness, dan Shapiro–Wilk. Distribusi nonnormal tidak otomatis memerlukan transformasi.
3. **Hubungan antarfitur:** pemeriksaan asosiasi/korelasi sesuai jenis fitur dan VIF pada fitur yang telah dikodekan. VIF diperiksa menggunakan data latih.
4. **Preprocessing:** pengkodean ulang indikator layanan yang berulang, `OneHotEncoder`, serta `StandardScaler` untuk model yang memerlukan skala fitur sebanding. `SimpleImputer` tersedia pada pipeline meskipun tidak ditemukan nilai kosong.
5. **Pemisahan data:** pembagian terstratifikasi 80:20.
6. **Benchmark:** `StratifiedKFold` lima fold pada data latih dengan pipeline dan preprocessor.
7. **Perbaikan model:** pengujian RandomOverSampler, kelompok tenure, dan jumlah layanan tambahan. Resampling dilakukan hanya pada bagian latih setiap fold.
8. **Tuning:** `GridSearchCV` pada model yang terpilih, diikuti pemeriksaan threshold berdasarkan prediksi out-of-fold data latih.
9. **Evaluasi akhir:** penggunaan `.fit()` pada data latih dan `.predict()` pada data uji, disertai confusion matrix dan metrik berbasis probabilitas.
10. **Simulasi bisnis:** perbandingan kampanye awal dan perbaikan, rincian biaya–manfaat, ROI, serta sensitivitas efektivitas retensi.

## 4. Key Insights

### Kualitas Data dan Generalisasi

- Tidak ditemukan missing value setelah pemeriksaan data.
- Terdapat 77 baris identik. Baris tersebut dipertahankan karena identitas pelanggan tidak tersedia untuk memastikan apakah merupakan duplikasi pelanggan yang sama.
- Fitur numerik tidak berdistribusi normal. Hal ini tidak menggugurkan penggunaan algoritma yang diuji; scaling dan regularisasi tetap digunakan sesuai kebutuhan model.
- `MonthlyCharges` memiliki VIF sekitar 11,34. Fitur dipertahankan untuk prediksi dengan regularisasi, tetapi interpretasi koefisien perlu berhati-hati.
- Decision Tree dan Random Forest pada konfigurasi awal menunjukkan kesenjangan besar antara F1 latih dan validasi, yang mengindikasikan overfitting.

### Logistic Regression Memimpin Benchmark Validasi Awal

| Algoritma | Rata-rata F1 validasi |
|---|---:|
| **Logistic Regression** | **0,580** |
| HistGradientBoosting | 0,568 |
| LightGBM | 0,567 |
| KNN | 0,544 |
| Random Forest | 0,528 |
| Decision Tree | 0,478 |

Logistic Regression dipilih untuk tahap perbaikan berdasarkan F1 validasi awal tertinggi. Kesimpulan ini berlaku untuk konfigurasi dan metode yang diuji, bukan klaim bahwa Logistic Regression selalu menjadi algoritma terbaik.

HistGradientBoosting mencapai F1 uji awal sekitar 0,633, sedikit di atas Logistic Regression setelah perbaikan sebesar 0,632. Selisih tersebut belum membuktikan keunggulan yang konsisten. HistGradientBoosting tetap menjadi kandidat pembanding pada data baru; seluruh algoritma belum menjalani tuning dengan cakupan yang sama.

### Perbaikan Meningkatkan Deteksi Churn dengan Konsekuensi Lebih Banyak Salah Sasaran

| Metrik pada 986 data uji | Model awal | Setelah perbaikan |
|---|---:|---:|
| F1-score | 0,594 | **0,632** |
| ROC AUC | 0,856 | 0,858 |
| Accuracy | 79,6% | 74,8% |
| Precision | 63,4% | 51,8% |
| Recall | 55,9% | **81,0%** |
| False-positive rate | 11,8% | 27,4% |
| True Positive | 147 | **213** |
| True Negative | 638 | 525 |
| False Positive | 85 | 198 |
| False Negative | 116 | **50** |

Model perbaikan mendeteksi **66 pelanggan churn tambahan**, tetapi juga menghasilkan **113 false positive tambahan**. Daftar calon kontak meningkat dari 232 menjadi 411 pelanggan. Model menjadi lebih sesuai untuk tujuan menjangkau calon churn, dengan kebutuhan kapasitas kontak dan anggaran yang lebih besar.

Sebagian besar peningkatan berasal dari persiapan data, terutama oversampling. Pada data uji, F1 meningkat sekitar **3,39 poin persentase** setelah persiapan dan **0,42 poin persentase** lagi setelah tuning. Tambahan feature engineering setelah oversampling relatif kecil pada validasi.

Konfigurasi Logistic Regression yang dipertahankan menggunakan `C=0.1`, penalti `l1`, solver `liblinear`, dan threshold **0,50**, bersama feature engineering dan oversampling. Tuning menguji lima nilai `C` dan dua jenis penalti. Class weight tidak ditambahkan pada konfigurasi dengan oversampling.

### Simulasi Biaya–Manfaat: Manfaat Bersih Naik, ROI Turun

**Konteks simulasi:** 100.000 pelanggan hipotetis dalam satu kampanye enam bulan. Hasil data uji diskalakan secara proporsional, dengan efektivitas pencegahan churn tambahan sebesar 20%. Kontribusi bulanan diasumsikan Rp0,00015 miliar per pelanggan, tambahan masa bertahan enam bulan, dan biaya kontak/penawaran Rp0,00005 miliar per pelanggan.

Referral diaktifkan dalam contoh utama: 20 pelanggan baru pada kampanye awal dan 30 pada kampanye perbaikan, dengan kontribusi bersih Rp0,0003 miliar per pelanggan baru. Asumsi tersebut belum dibuktikan dan dapat dinonaktifkan menggunakan `gunakan_referral = False`.

| Komponen, dalam Rp miliar | Model awal | Model perbaikan |
|---|---:|---:|
| Manfaat langsung | 2,709 | 3,928 |
| Manfaat tidak langsung | 0,006 | 0,009 |
| **Total manfaat** | **2,715** | **3,937** |
| Biaya langsung | 1,366 | 2,334 |
| Biaya tidak langsung | 0,034 | 0,055 |
| **Total biaya** | **1,400** | **2,389** |
| **Manfaat bersih** | **1,314** | **1,548** |
| ROI | 93,9% | 64,8% |

*Angka dibulatkan; total dihitung menggunakan nilai sebelum pembulatan.*

Tambahan manfaat bersih diperkirakan **Rp0,234 miliar**. Namun, ROI menurun karena kenaikan pengeluaran lebih besar secara proporsional. Manfaat bersih absolut dan efisiensi modal perlu dipertimbangkan bersama.

Tanpa manfaat referral, estimasi manfaat bersih menjadi sekitar Rp1,308 miliar untuk model awal dan Rp1,539 miliar untuk model perbaikan, dengan kenaikan sekitar **Rp0,231 miliar**.

![Perbandingan biaya dan manfaat](cost_benefit_comparison.png)

Grafik sensitivitas menunjukkan bahwa, dengan asumsi lainnya tetap dan referral aktif, model perbaikan menghasilkan manfaat bersih lebih besar ketika efektivitas pencegahan churn melebihi sekitar **16,1%**. Ini merupakan titik kesetaraan manfaat bersih antara kedua kampanye, bukan titik impas masing-masing kampanye terhadap kondisi tanpa kampanye.

![Sensitivitas efektivitas retensi](cost_benefit_sensitivity.png)

### Batasan Analisis

- Prediksi churn belum membuktikan bahwa kontak atau penawaran dapat mencegah churn. Efektivitas retensi dan referral memerlukan pengujian dengan kelompok kontrol.
- Nilai finansial adalah simulasi berdasarkan item imajiner, bukan penghematan atau pendapatan yang sudah terealisasi.
- Perusahaan belum memberikan biaya aktual, kapasitas operasional, periode prediksi churn, identitas pelanggan, atau tanggal observasi.
- Sebanyak 31 observasi data uji memiliki profil prediktor yang juga terdapat pada data latih. F1 sekitar 0,626 ketika profil tersebut dikeluarkan dalam pemeriksaan sensitivitas.
- Threshold dipilih berdasarkan F1, sehingga belum merupakan threshold yang memaksimalkan keuntungan.
- Perbandingan bisnis memakai kelompok pelanggan yang sama, tetapi jumlah kontak dan anggarannya berbeda. Evaluasi dengan kapasitas kontak yang sama merupakan langkah lanjutan.

## 5. Summary

Logistic Regression merupakan model pilihan berdasarkan benchmark validasi awal. Kombinasi penanganan ketimpangan kelas, feature engineering, dan tuning meningkatkan F1 serta kemampuan mendeteksi churn, dengan konsekuensi penurunan precision dan kenaikan jumlah kontak salah sasaran.

Simulasi bisnis menunjukkan potensi kenaikan manfaat bersih setelah seluruh komponen biaya dimasukkan. Hasilnya tetap bergantung pada efektivitas intervensi, kontribusi pelanggan, biaya kontak, dan asumsi operasional. Langkah berikutnya adalah menjalankan pilot retensi secara acak dengan kelompok kontrol, memperbarui asumsi menggunakan hasil aktual, dan membandingkan Logistic Regression dengan HistGradientBoosting pada data baru.

## Menjalankan Notebook

Simpan file berikut pada folder yang sama:

```text
README.md
telco_churn_rincian_biaya_manfaat.ipynb
data_telco_customer_churn.csv
requirements.txt
cost_benefit_comparison.png
cost_benefit_sensitivity.png
```

Instal dependensi:

```bash
pip install -r requirements.txt
```

Buka [`telco_churn_rincian_biaya_manfaat.ipynb`](telco_churn_rincian_biaya_manfaat.ipynb) dan jalankan sel secara berurutan. Bagian **15** membahas kontribusi kampanye sederhana; Bagian **16** merinci analisis biaya–manfaat penuh. Sesuaikan jumlah pelanggan, efektivitas retensi, tarif, dan pilihan referral pada kode untuk menguji skenario lain.

> Untuk GitHub, unggah README, notebook, requirements, dan gambar pendukung secara bersamaan agar tautan relatif dan grafik dapat ditampilkan.
