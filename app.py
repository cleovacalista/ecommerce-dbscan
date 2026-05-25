import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# Konfigurasi Halaman Utama Dashboard
st.set_page_config(
    page_title="DBSCAN E-Commerce Clustering",
    page_icon="🛍️",
    layout="wide"
)

# Judul Utama Dashboard
st.title("📊 Klasterisasi Pola Belanja Pelanggan E-Commerce")
st.caption("Aplikasi Analisis Segmentasi Pasar Menggunakan Algoritma DBSCAN Berbasis Kerangka KDD")
st.markdown("---")

# Fungsi Caching untuk Memuat dan Memproses Data Menggunakan Kerangka KDD
@st.cache_data
def load_and_process_kdd_data(file_path):
    # 1. Tahap Selection & Preprocessing
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    df = df.dropna(subset=['CustomerID'])
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    
    # 2. Tahap Transformation (Ekstraksi Fitur RFM)
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    # Transformasi Logaritmik untuk Mengatasi Skewness Data Ritel
    rfm_log = rfm[['Recency', 'Frequency', 'Monetary']].apply(np.log1p)
    
    # Standardisasi Skala Jarak Fitur
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    
    return rfm, rfm_scaled

# Memuat Dataset
try:
    rfm_data, rfm_scaled_features = load_and_process_kdd_data('data.csv')
except FileNotFoundError:
    st.error("❌ Berkas 'data.csv' tidak ditemukan! Pastikan posisi file sudah berada di direktori proyek Anda.")
    st.stop()

# Pembuatan Sidebar untuk Kontrol Parameter Pemodelan
st.sidebar.header("🎛️ Pengaturan Parameter DBSCAN")
st.sidebar.markdown("Sesuaikan nilai parameter kepadatan spasial di bawah ini:")

epsilon = st.sidebar.slider("Nilai Epsilon (Eps)", min_value=0.1, max_value=1.5, value=0.5, step=0.05)
min_pts = st.sidebar.number_input("Minimum Points (MinPts)", min_value=2, max_value=50, value=5, step=1)

# 3. Tahap Data Mining (Proses Pemodelan DBSCAN)
dbscan_model = DBSCAN(eps=epsilon, min_samples=min_pts)
cluster_labels = dbscan_model.fit_predict(rfm_scaled_features)
rfm_data['Cluster'] = cluster_labels

# Menghitung Metrik Evaluasi Kualitas Klaster
total_customers = len(rfm_data)
unique_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
noise_counts = list(cluster_labels).count(-1)

# Tampilan Metrik Ringkasan di Atas Ringkasan Eksekutif
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("Total Pelanggan Teranalisis", f"{total_customers:,} ID")
with col_m2:
    st.metric("Jumlah Klaster Alami", f"{unique_clusters} Klaster")
with col_m3:
    st.metric("Deteksi Outliers (Noise)", f"{noise_counts} Pelanggan")
with col_m4:
    # Menghitung Silhouette Score secara dinamis jika klaster valid terbentuk
    if unique_clusters >= 1:
        score = silhouette_score(rfm_scaled_features, cluster_labels)
        st.metric("Silhouette Score", f"{score:.4f}")
    else:
        st.metric("Silhouette Score", "N/A")

st.markdown("---")

# Layout Komponen Utama: Visualisasi vs Tabel Profiling
col_graph, col_table = st.columns([3, 2])

with col_graph:
    st.subheader("🌐 Visualisasi Sebaran Spasial Klaster 3D")
    # Mapping nama klaster ke representasi string agar visualisasi lebih informatif
    rfm_data['Cluster_Name'] = rfm_data['Cluster'].apply(lambda x: "VIP Wholesale (Noise)" if x == -1 else f"Klaster {x}")
    
    fig = px.scatter_3d(
        rfm_data, 
        x='Recency', 
        y='Frequency', 
        z='Monetary',
        color='Cluster_Name',
        log_x=True, log_y=True, log_z=True,
        title="Distribusi Pola RFM Konsumen (Skala Logaritmik)",
        labels={'Cluster_Name': 'Segmen Pelanggan'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig, use_container_width=True)

with col_table:
    st.subheader("📋 Tabel Profiling Rata-Rata Pola Belanja")
    profile_summary = rfm_data.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'CustomerID': 'count'
    }).rename(columns={'CustomerID': 'Jumlah Pelanggan'}).reset_index()
    
    # Memformat tampilan nominal mata uang dan angka pecahan desimal
    profile_summary['Recency'] = profile_summary['Recency'].map("{:,.1f} Hari".format)
    profile_summary['Frequency'] = profile_summary['Frequency'].map("{:,.1f} Kali".format)
    profile_summary['Monetary'] = profile_summary['Monetary'].map("${:,.2f}".format)
    
    st.dataframe(profile_summary, use_container_width=True, hide_index=True)

st.markdown("---")

# 4. Tahap Interpretation: Rekomendasi Strategi Bisnis Ritel Terarah
st.subheader("💡 Interpretasi Karakteristik Segmen dan Strategi Bisnis")
col_s1, col_s2, col_s3 = st.columns(3)

with col_s1:
    st.info("**Klaster 0: Pelanggan Aktif & Loyal**\n\n"
            "• **Profil**: Interaksi transaksi tergolong baru, frekuensi belanja stabil, serta kontribusi keuangan sehat.\n"
            "• **Strategi Bisnis**: Terapkan program poin loyalitas (*loyalty reward*) dan berikan akses eksklusif untuk produk baru sebelum dirilis ke pasar umum.")

with col_s2:
    st.warning("**Klaster 1: Pelanggan Hibernasi / Pasif**\n\n"
               "• **Profil**: Nilai jeda transaksi terakhir sangat lama, frekuensi kunjungan rendah dengan pengeluaran minim.\n"
               "• **Strategi Bisnis**: Luncurkan program *re-activation email marketing*, kirimkan voucer diskon khusus pembelanjaan kembali, atau survei kepuasan pelanggan.")

with col_s3:
    st.error("**Klaster -1: Outliers / VIP Wholesale**\n\n"
             "• **Profil**: Kelompok belanja bernilai ekstrem masif. Kuantitas transaksi sangat besar di luar batas konsumen retail biasa.\n"
             "• **Strategi Bisnis**: Alihkan penanganan ke skema kemitraan korporasi khusus B2B, berikan manajer akun personal (*dedicated account manager*), serta sistem pemotongan harga grosir langsung.")