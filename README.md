# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan - Jaya Jaya Institut

## Business Understanding
Jaya Jaya Institut merupakan institusi pendidikan tinggi yang telah berdiri sejak tahun 2000. Saat ini, institusi menghadapi tantangan besar terkait tingginya angka mahasiswa yang tidak menyelesaikan studi (*dropout*). Hal ini menjadi perhatian utama karena kelangsungan operasional dan reputasi institusi sangat bergantung pada keberhasilan mahasiswa dalam menyelesaikan pendidikan mereka.

### Permasalahan Bisnis
Tingginya angka *dropout* di Jaya Jaya Institut menimbulkan beberapa masalah utama:
* **Penurunan Akreditasi:** Angka kelulusan mahasiswa adalah parameter vital dalam penilaian akreditasi nasional dan internasional.
* **Kerugian Finansial:** Setiap mahasiswa yang *dropout* berarti kehilangan potensi pendapatan dari biaya kuliah (UKT) yang seharusnya berkelanjutan hingga lulus.
* **Reputasi Institusi:** Calon mahasiswa baru dan masyarakat cenderung menilai kualitas institusi berdasarkan rasio kelulusannya.
* **Inakurasi Monitoring:** Pihak manajemen belum memiliki alat otomatis untuk mendeteksi secara dini mahasiswa yang berisiko *dropout*, sehingga intervensi seringkali terlambat dilakukan.

### Cakupan Proyek
Cakupan pengerjaan proyek ini meliputi:
* Melakukan analisis data untuk mengidentifikasi faktor-faktor yang berkorelasi kuat dengan status kelulusan mahasiswa.
* Membangun model *Machine Learning* klasifikasi biner untuk memprediksi potensi *dropout* (Lulus vs Dropout).
* Mengembangkan dashboard bisnis interaktif untuk memonitor data mahasiswa secara *real-time*.
* Menyediakan rekomendasi kebijakan berbasis data untuk menekan angka *dropout*.

## Persiapan

### Sumber data:
Dataset berasal dari database internal Jaya Jaya Institut yang berisi 4.424 baris data dengan 37 atribut, mencakup data demografi, ekonomi, dan performa akademik mahasiswa pada semester 1 dan semester 2.

### Setup environment:
Proyek ini menggunakan bahasa Python dengan pustaka sebagai berikut:
* `pandas`, `numpy` (Manajemen data)
* `scikit-learn` (Pemodelan Machine Learning)
* `matplotlib`, `seaborn`, `plotly` (Visualisasi data)
* `streamlit` (Dashboard & Deployment)
* `joblib` (Penyimpanan model)

Untuk menginstal semua *dependency*, jalankan perintah:
```bash
pip install -r requirements.txt
```

## Business Dashboard
*Business Dashboard* dikembangkan menggunakan Streamlit untuk mempermudah manajemen dalam melihat tren data. Dashboard ini memiliki beberapa fitur:
* **Filter Interaktif:** Memungkinkan pengguna memfilter data berdasarkan Gender dan Status Beasiswa.
* **Analisis Proporsi:** Visualisasi persentase mahasiswa berdasarkan status kelulusan.
* **Analisis Faktor Risiko:** Grafik yang membandingkan performa akademik semester 2 terhadap kemungkinan *dropout*.
Link Dashboard: https://abdurrkhmn-data-science-for-education-solutions--akh-app-u7phtc.streamlit.app/

## Menjalankan Sistem Machine Learning
Untuk menjalankan prototipe sistem *machine learning* secara lokal, ikuti langkah berikut:
1. Pastikan file `model_dropout.pkl`, `app.py`, dan `data.csv` berada dalam satu folder yang sama.
2. Buka terminal atau command prompt.
3. Jalankan perintah:
   ```bash
   streamlit run app.py
   ```
4. Masukkan data mahasiswa pada form yang tersedia di halaman "Prediksi Kelulusan".
5. Klik tombol "Analisis" untuk mendapatkan hasil prediksi apakah mahasiswa tersebut berpotensi Lulus atau *Dropout*.

## Conclusion
Berdasarkan hasil analisis, ditemukan bahwa performa akademik pada tahun pertama (khususnya semester 2) merupakan faktor penentu paling signifikan. Mahasiswa dengan jumlah unit yang disetujui rendah dan memiliki tunggakan biaya kuliah memiliki risiko *dropout* yang jauh lebih tinggi. Model *Machine Learning* yang dibangun telah divalidasi mampu mengklasifikasikan risiko ini dengan akurasi yang baik, sehingga dapat digunakan sebagai sistem peringatan dini.

## Rekomendasi Action Items
Berikut adalah beberapa rekomendasi untuk Jaya Jaya Institut:
* **Early Warning System (EWS):** Mengintegrasikan model machine learning ini ke dalam sistem informasi akademik untuk memberikan notifikasi otomatis kepada dosen pembimbing jika mahasiswa terdeteksi berisiko tinggi di akhir semester 1.
* **Program Bantuan Finansial Terfokus:** Memberikan skema cicilan khusus atau bantuan dana darurat bagi mahasiswa yang teridentifikasi sebagai penunggak biaya (*Debtor*) namun memiliki prestasi akademik yang baik.
* **Mentoring Akademik Intensif:** Mengadakan program pendampingan khusus bagi mahasiswa yang memiliki jumlah unit lulus di bawah standar pada semester pertama untuk mencegah kegagalan permanen di semester berikutnya.

---
