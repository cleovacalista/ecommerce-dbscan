# Klasterisasi Pola Belanja Pelanggan E-Commerce dengan Algoritma DBSCAN

Aplikasi *web dashboard* interaktif berbasis sains data ini dibuat untuk mendeteksi segmen perilaku pembelanjaan pelanggan retail online berdasarkan fitur **Recency, Frequency, dan Monetary (RFM)** melalui kerangka kerja **KDD (Knowledge Discovery in Databases)**.

## Identitas Pengembang
* **Nama**: Cleova Calista Aziza Kayviar
* **NIM**: 23.11.5532
* **Instansi**: Universitas AMIKOM Yogyakarta
* **Tahun Akurasi Proyek**: 2026

## Fitur Utama Aplikasi
1. **Prapemrosesan Otomatis**: Pembersihan data transaksi retur/negatif dan penanganan nilai kosong.
2. **Transformasi Skala**: Log transformasi bersama standarisasi *StandardScaler* untuk optimalisasi perhitungan jarak Euclidean algoritma DBSCAN.
3. **Simulasi Dinamis**: Pengaturan parameter spasial $\epsilon$ (*Epsilon*) dan *Minimum Points* secara langsung lewat antarmuka grafis.
4. **Visualisasi Interaktif**: Plot spasial 3D interaktif yang didukung pustaka Plotly Engine.