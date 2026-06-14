import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from scipy.spatial import distance_matrix

# Memastikan Matplotlib berjalan stabil tanpa GUI di server cloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

# ==========================================
# 1. KONFIGURASI HALAMAN & CSS DASHBOARD
# ==========================================
st.set_page_config(
    page_title="Mikrozonasi Seismik HVSR - UIN Sunan Kalijaga",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-title { font-size: 32px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
    .sub-title { font-size: 18px; color: #4B5563; margin-bottom: 20px; }
    .section-title { font-size: 24px; font-weight: bold; color: #0D9488; margin-top: 25px; margin-bottom: 15px; border-bottom: 2px solid #0D9488; padding-bottom: 5px; }
    .metric-box { background-color: #F3F4F6; padding: 15px; border-radius: 10px; border-left: 5px solid #1E3A8A; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. FUNGSI PERHITUNGAN & INTERPOLASI
# ==========================================
def calculate_kg(a0, f0):
    return (a0 ** 2) / f0

def classify_kg(kg):
    if kg < 3: return "Rendah"
    elif 3 <= kg <= 6: return "Menengah"
    else: return "Tinggi"

def classify_soil(f0):
    if f0 > 10: return "Klas I (Batuan Keras)"
    elif 4 <= f0 <= 10: return "Klas II (Tanah Padat)"
    elif 1 <= f0 < 4: return "Klas III (Sedimen Sedang)"
    else: return "Klas IV (Sedimen Lunak)"

def color_picker_kg(status):
    if status == "Rendah": return "green"
    elif status == "Menengah": return "orange"
    else: return "red"

def idw_interpolation(x, y, z, xi, yi, power=2):
    points = np.vstack((x, y)).T
    grid_points = np.vstack((xi.ravel(), yi.ravel())).T
    dist = distance_matrix(points, grid_points)
    dist = np.where(dist == 0, 1e-12, dist)
    weights = 1.0 / (dist ** power)
    weights /= np.sum(weights, axis=0)
    zi = np.dot(z, weights)
    return zi.reshape(xi.shape)

# REPLIKA ENGINE SURFER & QGIS: Tanpa Titik Koordinat Aksis & Bingkai Putih
def create_surfer_contour_overlay(xi, yi, zi, cmap_name, levels=15):
    # Menggunakan mode tight layout agar frame putih bawaan hilang total
    fig = plt.figure(figsize=(10, 10), dpi=300, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')  # Mematikan garis koordinat X, Y, dan label angka aksis
    
    # 1. Filled Contours (Warna Gradasi Padat Surfer Style)
    ax.contourf(xi, yi, zi, levels=levels, cmap=cmap_name)
    
    # 2. Contour Lines (Garis batas kontur hitam tipis penegas anomali wilayah)
    ax.contour(xi, yi, zi, levels=levels, colors='black', linewidths=0.5, alpha=0.7)
    
    # Simpan plot ke memori sebagai format PNG Transparan tanpa whitespace sisa
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    img_data = plt.imread(buf)
    plt.close(fig)
    return img_data

# Inisialisasi Session State Data
if 'df_data' not in st.session_state:
    st.session_state.df_data = None

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #1E3A8A;'>🕌 GEOFISIKA</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; margin-top: 0px; color: #4B5563; font-size: 14px;'>UIN SUNAN KALIJAGA</h3>", unsafe_allow_html=True)
    st.markdown("---")
    menu = st.radio("Pilih Menu Dashboard:", ["Analisis Kerentanan", "Mikrozonasi Spasial", "Analisis Resonansi"])
    st.markdown("---")
    st.caption("UAS Seismologi - © 2026")

# ==========================================
# 4. HEADER UTAMA & UPLOAD DATA
# ==========================================
st.markdown('<div class="main-title">Aplikasi Mikrozonasi Seismik HVSR</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Program Studi Geofisika - UIN Sunan Kalijaga Yogyakarta</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Unggah File CSV Pengukuran Mikrotremor:", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
        df.columns = df.columns.str.strip().str.lower().str.replace('fo', 'f0').str.replace('ao', 'a0')
        required = ['titik', 'longitude', 'latitude', 'f0', 'a0']
        
        if all(col in df.columns for col in required):
            df = df.rename(columns={'titik': 'Titik', 'longitude': 'Longitude', 'latitude': 'Latitude', 'f0': 'f0', 'a0': 'A0'})
            for c in ['f0', 'A0', 'Latitude', 'Longitude']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df = df.dropna(subset=['f0', 'A0', 'Latitude', 'Longitude'])
            
            df['Kg'] = calculate_kg(df['A0'], df['f0'])
            df['Tingkat Kerentanan'] = df['Kg'].apply(classify_kg)
            df['Karakteristik Tanah'] = df['f0'].apply(classify_soil)
            st.session_state.df_data = df
    except Exception as e:
        st.error(f"Gagal memproses file: {e}")

df = st.session_state.df_data

# ==========================================
# MENU MAIN FLOW
# ==========================================
if menu == "Analisis Kerentanan":
    if df is None:
        st.warning("Silakan unggah data lapangan CSV terlebih dahulu.")
    else:
        st.markdown('<div class="section-title">Ringkasan Statistik Lapisan</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.markdown(f'<div class="metric-box"><b>Rata-rata f0</b><h2>{df["f0"].mean():.2f} Hz</h2></div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-box"><b>Rata-rata A0</b><h2>{df["A0"].mean():.2f}</h2></div>', unsafe_allow_html=True)
        col3.markdown(f'<div class="metric-box"><b>Rata-rata Kg</b><h2>{df["Kg"].mean():.2f}</h2></div>', unsafe_allow_html=True)
        st.write("")
        st.dataframe(df[['Titik', 'Longitude', 'Latitude', 'f0', 'A0', 'Kg', 'Tingkat Kerentanan', 'Karakteristik Tanah']], use_container_width=True)

elif menu == "Mikrozonasi Spasial":
    if df is None:
        st.warning("Silakan unggah data lapangan CSV terlebih dahulu.")
    else:
        st.markdown('<div class="section-title">Output Peta Mikrozonasi Spasial (Google Satellite)</div>', unsafe_allow_html=True)
        
        # Penentuan batas koordinat area secara ketat di area pengukuran
        center_lat, center_lon = df['Latitude'].mean(), df['Longitude'].mean()
        pad = 0.002  # Ukuran kotak pembatas area zonasi agar pas dengan sebaran data
        min_lat, max_lat = df['Latitude'].min() - pad, df['Latitude'].max() + pad
        min_lon, max_lon = df['Longitude'].min() - pad, df['Longitude'].max() + pad
        bounds = [[min_lat, min_lon], [max_lat, max_lon]]
        
        # Grid Interpolasi IDW
        x, y = df['Longitude'].values, df['Latitude'].values
        x_line = np.linspace(min_lon, max_lon, 250)
        y_line = np.linspace(min_lat, max_lat, 250)
        xi, yi = np.meshgrid(x_line, y_line)
        
        zi_a0 = idw_interpolation(x, y, df['A0'].values, xi, yi)
        zi_f0 = idw_interpolation(x, y, df['f0'].values, xi, yi)
        zi_kg = idw_interpolation(x, y, df['Kg'].values, xi, yi)
        
        # Pembuatan Gambar Raster Kontur Padat Murni Bersih (Tanpa Teks Sumbu/Aksis Koordinat)
        img_a0 = create_surfer_contour_overlay(xi, yi, zi_a0, 'rainbow') # Skema spektrum QGIS/Surfer
        img_f0 = create_surfer_contour_overlay(xi, yi, zi_f0, 'viridis')
        img_kg = create_surfer_contour_overlay(xi, yi, zi_kg, 'jet')     # Skema Jet andalan Surfer
        
        # --- FUNGSI UTAMA GENERATOR INTERAKTIF FOLIUM ---
        def generate_mikrozonasi_map(overlay_img=None, is_peta_1=False):
            # Inisialisasi basemap murni Google Satellite untuk semua nomor peta
            m = folium.Map(
                location=[center_lat, center_lon], zoom_start=16,
                tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                attr='Google Satellite Imagery',
                control_scale=True  # ⚡ BARIS INI YANG MENAMBAHKAN SCALE BAR OTOMATIS (Meter / Miles)
            )
            
            # Jika Peta 2, 3, atau 4 (overlay_img diisi), tempelkan raster konturnya ke peta
            if overlay_img is not None:
                folium.raster_layers.ImageOverlay(
                    image=overlay_img,
                    bounds=bounds,
                    opacity=0.60,  # Nilai transparansi agar rona satelit di bawah kontur tetap terlihat
                    mercator_project=True
                ).add_to(m)
            
# Modifikasi agar titik lingkaran HANYA muncul di Peta 1
            if is_peta_1:
                for _, row in df.iterrows():
                    popup_html = f"""
                    <div style='font-family: Arial, sans-serif; font-size: 12px; width: 140px;'>
                        <h5 style='margin:0 0 5px 0; color:#1E3A8A; border-bottom:1px solid #CCC; padding-bottom:3px;'><b>Titik Ke: {row['Titik']}</b></h5>
                        <b>Latitude:</b> {row['Latitude']:.5f}<br>
                        <b>Longitude:</b> {row['Longitude']:.5f}
                    </div>
                    """
                    folium.CircleMarker(
                        location=[row['Latitude'], row['Longitude']], radius=5,
                        popup=folium.Popup(popup_html, max_width=180),
                        color='black', weight=1.2,
                        fill=True, fill_color=color_picker_kg(row['Tingkat Kerentanan']), fill_opacity=1.0
                    ).add_to(m)
                
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']], radius=5,
                    popup=folium.Popup(popup_html, max_width=180),
                    color='black', weight=1.2,
                    fill=True, fill_color=color_picker_kg(row['Tingkat Kerentanan']), fill_opacity=1.0
                ).add_to(m)
            return m

        # Pembagian Output ke dalam 4 Modul Tab Navigasi Berurutan
        tab1, tab2, tab3, tab4 = st.tabs([
            "📌 Peta 1: Kerentanan Seismik (Titik Koordinat)", 
            "🟢 Peta 2: Peta Kontur Amplifikasi ($A_0$)", 
            "🟣 Peta 3: Peta Kontur Frekuensi Dominan ($f_0$)", 
            "🔴 Peta 4: Peta Kontur Indeks Kerentanan ($K_g$)"
        ])
        
        with tab1:
            st.markdown("#### Peta 1: Sebaran Spasial Titik Pengukuran Lapangan")
            st.caption("💡 Petunjuk: Klik pada bulatan titik stasiun untuk memeriksa data letak posisi geografisnya (Titik ke-n beserta koordinat Lintang/Bujurnya).")
            st_folium(generate_mikrozonasi_map(overlay_img=None, is_peta_1=True), width=1100, height=520, key="peta_1_koordinat")
            
        with tab2:
            st.markdown("#### Peta 2: Visualisasi Kontur Padat Faktor Amplifikasi Situs ($A_0$)")
            st.caption("✨ Tampilan raster kontur mulus (borderless) tanpa teks aksis koordinat bawaan.")
            st_folium(generate_mikrozonasi_map(img_a0, is_peta_1=False), width=1100, height=520, key="peta_2_a0_clean")
            
        with tab3:
            st.markdown("#### Peta 3: Visualisasi Kontur Padat Frekuensi Dominan Tanah ($f_0$)")
            st.caption("✨ Tampilan raster kontur mulus (borderless) tanpa teks aksis koordinat bawaan.")
            st_folium(generate_mikrozonasi_map(img_f0, is_peta_1=False), width=1100, height=520, key="peta_3_f0_clean")
            
        with tab4:
            st.markdown("#### Peta 4: Visualisasi Kontur Padat Indeks Kerentanan Seismik ($K_g$)")
            st.caption("✨ Tampilan raster kontur mulus (borderless) tanpa teks aksis koordinat bawaan.")
            st_folium(generate_mikrozonasi_map(img_kg, is_peta_1=False), width=1100, height=520, key="peta_4_kg_clean")
            
elif menu == "Analisis Resonansi":
    if df is None:
        st.warning("Silakan unggah data lapangan CSV terlebih dahulu.")
    else:
        st.markdown('<div class="section-title">Analisis Bahaya Resonansi Struktur</div>', unsafe_allow_html=True)
        num_floors = st.number_input("Masukkan Jumlah Lantai Bangunan (N):", min_value=1, max_value=30, value=3)
        fb = 10.0 / num_floors
        st.info(f"Frekuensi Alami Bangunan Estimasian ($f_b$): {fb:.2f} Hz")
