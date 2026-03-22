# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan - Jaya Jaya Institut

## Business Understanding
Jaya Jaya Institut merupakan institusi pendidikan tinggi yang telah berdiri sejak tahun 2000. Saat ini, institusi menghadapi tantangan besar terkait tingginya angka mahasiswa yang tidak menyelesaikan studi (*dropout*). Hal ini menjadi perhatian utama karena kelangsungan operasional dan reputasi institusi sangat bergantung pada keberhasilan mahasiswa dalam menyelesaikan pendidikan mereka.

### Permasalahan Bisnis
Tingginya angka *dropout* di Jaya Jaya Institut menimbulkan beberapa masalah utama:
* **Penurunan Akreditasi:** Angka kelulusan mahasiswa adalah parameter vital dalam penilaian akreditasi nasional dan internasional.
* **Kerugian Finansial:** Setiap mahasiswa yang *dropout* berarti kehilangan potensi pendapatan dari biaya kuliah (UKT).
* **Reputasi Institusi:** Masyarakat menilai kualitas institusi berdasarkan rasio kelulusannya.
* **Inakurasi Monitoring:** Belum adanya alat otomatis untuk mendeteksi dini mahasiswa berisiko *dropout*.

### Cakupan Proyek
* Analisis korelasi faktor terhadap status kelulusan.
* Pembangunan model *Machine Learning* klasifikasi biner (Lulus vs Dropout).
* Pengembangan dashboard bisnis interaktif menggunakan Streamlit.
* Pemberian rekomendasi kebijakan berbasis data.

## Persiapan Proyek

### 1. Sumber Data
Dataset berasal dari database internal Jaya Jaya Institut yang terdiri dari 4.424 baris data dengan 37 atribut. Secara publik, referensi dataset ini merujuk pada:
* **Link Dataset:** [UCI Machine Learning Repository - Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

### 2. Membuat dan Mengaktifkan Virtual Environment (venv)
Untuk menjaga stabilitas lingkungan pengembangan, sangat disarankan untuk menggunakan *virtual environment*.

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalasi Library (Dependencies)
Pastikan *virtual environment* telah aktif, kemudian jalankan perintah berikut:
```bash
pip install -r requirements.txt
```

## Business Dashboard
*Business Dashboard* dikembangkan untuk mempermudah manajemen dalam memantau tren data secara *real-time*. 
* **Link Dashboard Online:** [Jaya Jaya Institut Analytics](https://abdurrkhmn-data-science-for-education-solutions--akh-app-u7phtc.streamlit.app/)


**Fitur Dashboard:**
* **Filter Interaktif:** Berdasarkan Gender dan Status Beasiswa.
* **Distribusi Usia & Status:** Visualisasi hubungan usia pendaftaran dengan hasil studi.
* **Analisis Nilai:** Perbandingan nilai semester 2 terhadap status kelulusan.

## Menjalankan Sistem Machine Learning (Lokal)
1. Pastikan file `model_dropout.pkl`, `app.py`, `data.csv`, dan `requirements.txt` berada dalam satu folder.
2. Aktifkan *virtual environment* Anda.
3. Jalankan perintah:
   ```bash
   streamlit run app.py
   ```
4. Akses halaman **"Prediksi Kelulusan"** untuk mencoba input data mahasiswa.

## Conclusion
Performa akademik pada tahun pertama (khususnya semester 2) merupakan faktor penentu paling signifikan. Mahasiswa dengan jumlah unit yang disetujui rendah serta memiliki kendala finansial (*Debtor*) memiliki risiko *dropout* yang jauh lebih tinggi. Model *Machine Learning* yang dibangun menggunakan Random Forest telah divalidasi mampu mengklasifikasikan risiko ini dengan akurasi yang baik.

## Rekomendasi Action Items
* **Early Warning System (EWS):** Integrasi model ke sistem akademik untuk notifikasi dini kepada dosen pembimbing.
* **Bantuan Finansial Terfokus:** Skema cicilan khusus bagi mahasiswa *Debtor* dengan prestasi akademik baik.
* **Mentoring Intensif:** Pendampingan khusus bagi mahasiswa dengan unit lulus di bawah standar pada semester 1.