import streamlit as st
import pandas as pd

# Konfigurasi Halaman Utama Dashboard
st.set_page_config(
    page_title="DBSCAN E-Commerce Clustering",
    page_icon="🛍️",
    layout="wide"
)

# Judul Utama Dashboard
st.title("📊 Klasterisasi Pola Belanja Pelanggan E-Commerce")
st.caption("Hasil Analisis Segmentasi Pasar Menggunakan Algoritma DBSCAN (Pre-computed dari Google Colab)")
st.markdown("---")

# Menampilkan Metrik Evaluasi Kualitas Klaster (Berdasarkan Laporan UTS)
st.subheader("📈 Ringkasan Performa Model")
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.metric("Total Pelanggan Teranalisis", "4.338 ID") # Sesuaikan angkanya jika berbeda
with col_m2:
    st.metric("Jumlah Klaster Utama", "2 Klaster")
with col_m3:
    st.metric("Deteksi Outliers (Noise)", "62 Pelanggan")
with col_m4:
    st.metric("Silhouette Score", "0.2662")

st.markdown("---")

# Layout Komponen Utama: Menampilkan Gambar dari Colab
st.subheader("🌐 Visualisasi Hasil Klasterisasi")

try:
    # Menggunakan use_container_width=True (use_column_width sudah usang/deprecated di Streamlit terbaru)
    st.image(
        "grafik_3d.png", 
        caption="Distribusi Pola RFM Konsumen (Skala Logaritmik) - Output Google Colab", 
        use_container_width=True
    )
except FileNotFoundError:
    st.warning("⚠️ Gambar 'grafik_3d.png' belum dimasukkan ke direktori proyek Anda.")
except Exception as e:
    # Menangkap error lain (misal: file gambar korup / UnidentifiedImageError)
    st.error(f"❌ File gambar ditemukan, tetapi gagal dimuat. Detail error: {e}")

# 4. Tahap Interpretation: Rekomendasi Strategi Bisnis Ritel Terarah
st.subheader("💡 Interpretasi Karakteristik Segmen dan Strategi Bisnis")

# Tabel Ringkasan (Opsional, kamu bisa ganti dengan st.image("tabel_profil.png") jika tabelnya berupa gambar dari Colab)
data_tabel = {
    "Klaster": ["0", "1", "-1 (Outliers)"],
    "Recency": ["59 Hari", "156 Hari", "54 Hari"],
    "Frequency": ["5,3 kali", "1,0 kali", "33,2 kali"],
    "Monetary": ["$2.029", "$365", "$43.874"],
    "Karakteristik": ["Pelanggan Aktif & Loyal", "Pelanggan Pasif / Hibernasi", "VIP / Grosir Massal"]
}
st.table(pd.DataFrame(data_tabel))

col_s1, col_s2, col_s3 = st.columns(3)

with col_s1:
    st.info("**Klaster 0: Pelanggan Aktif & Loyal**\n\n"
            "• **Strategi Bisnis**: Terapkan program poin loyalitas (*loyalty reward*) dan berikan akses eksklusif untuk produk baru sebelum dirilis ke pasar umum.")

with col_s2:
    st.warning("**Klaster 1: Pelanggan Hibernasi / Pasif**\n\n"
               "• **Strategi Bisnis**: Luncurkan program *re-activation email marketing*, kirimkan voucer diskon khusus pembelanjaan kembali, atau survei kepuasan pelanggan.")

with col_s3:
    st.error("**Klaster -1: Outliers / VIP Wholesale**\n\n"
             "• **Strategi Bisnis**: Alihkan penanganan ke skema kemitraan korporasi khusus B2B, berikan manajer akun personal, serta sistem pemotongan harga grosir langsung.")