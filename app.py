import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io

# ==========================================
# 1. KONFIGURASI HALAMAN UTAMA & LAYOUT
# ==========================================
st.set_page_config(
    page_title="Mikrozonasi Seismik HVSR - UIN Sunan Kalijaga",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Kustomisasi CSS Dashboard Geofisika UIN
st.markdown("""
    <style>
    .main-title { font-size: 32px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
    .sub-title { font-size: 18px; color: #4B5563; margin-bottom: 20px; }
    .section-title { font-size: 24px; font-weight: bold; color: #0D9488; margin-top: 25px; margin-bottom: 15px; border-bottom: 2px solid #0D9488; padding-bottom: 5px; }
    .metric-box { background-color: #F3F4F6; padding: 15px; border-radius: 10px; border-left: 5px solid #1E3A8A; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    .ref-box { background-color: #EFF6FF; padding: 15px; border-radius: 8px; border-left: 5px solid #3B82F6; margin-top: 15px; font-size: 13px; color: #1E3A8A; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. FUNGSI GEOFISIKA & INTERPOLASI SPASIAL
# ==========================================
def calculate_kg(a0, f0):
    return (a0 ** 2) / f0

def classify_kg(kg):
    if kg < 3: return "Rendah"
    elif 3 <= kg <= 6: return "Menengah"
    else: return "Tinggi"

def classify_soil(f0):
    if f0 > 10: return "Klas I (Batuan Keras / Hard Rock)"
    elif 4 <= f0 <= 10: return "Klas II (Tanah Padat / Aluvium Ringan)"
    elif 1 <= f0 < 4: return "Klas III (Sedimen Sedang / Aluvium Tua)"
    else: return "Klas IV (Sedimen Tebal / Aluvium Lunak)"

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

# Helper untuk mengubah matriks kontur Matplotlib menjadi Gambar Overlay Folium
def create_contour_overlay(xi, yi, zi, cmap_name):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.position = [0, 0, 1, 1]
    ax.contourf(xi, yi, zi, levels=15, cmap=cmap_name)
    ax.contour(xi, yi, zi, levels=15, colors='black', linewidths=0.5)
    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True, dpi=200)
    buf.seek(0)
    plt.close(fig)
    return plt.imread(buf)

# Inisialisasi Session State Data
if 'df_data' not in st.session_state:
    st.session_state.df_data = None

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    try:
        st.image("logo_uin.png", use_container_width=True)
    except:
        st.markdown("<h2 style='text-align: center; color: #1E3A8A;'>🕌 UIN SUKA</h2>", unsafe_allow_html=True)
            
    st.markdown("<h3 style='text-align: center; margin-top: 10px; color: #111827; font-size: 16px;'>GEOFISIKA UIN SUKA</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    menu = st.radio(
        "Pilih Menu Dashboard:",
        ["Analisis Kerentanan", "Mikrozonasi Spasial", "Analisis Resonansi"]
    )
    st.markdown("---")
    st.caption("UAS Seismologi - © 2026")

# ==========================================
# 4. HEADER INTEGRASI HOME & INPUT DATA DIRECT
# ==========================================
st.markdown('<div class="main-title">Aplikasi Mikrozonasi Seismik HVSR</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Program Studi Geofisika - UIN Sunan Kalijaga Yogyakarta</div>', unsafe_allow_html=True)

if st.session_state.df_data is None:
    st.info("👋 Selamat Datang! Silakan unggah file CSV hasil pengukuran lapangan Anda di bawah ini untuk memulai pemetaan spasial dan analisis data.")

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
            st.success("Data CSV Lapangan Berhasil Diproses!")
        else:
            st.error("Format kolom CSV salah! Pastikan file memiliki nama kolom: TITIK, LONGITUDE, LATITUDE, F0, A0")
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")

df = st.session_state.df_data

# ==========================================
# MENU 1: ANALISIS KERENTANAN
# ==========================================
if menu == "Analisis Kerentanan":
    if df is None:
        st.warning("Silakan unggah file CSV data lapangan Anda terlebih dahulu pada panel di atas.")
    else:
        st.markdown('<div class="section-title">Ringkasan Statistik Parameter Struktur Lapisan</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1: 
            st.markdown(f'<div class="metric-box"><b>Rata-rata f0 (Frekuensi Dominan)</b><h2>{df["f0"].mean():.2f} Hz</h2></div>', unsafe_allow_html=True)
        with col2: 
            st.markdown(f'<div class="metric-box"><b>Rata-rata A0 (Amplifikasi)</b><h2>{df["A0"].mean():.2f}</h2></div>', unsafe_allow_html=True)
        with col3: 
            st.markdown(f'<div class="metric-box"><b>Rata-rata Kg (Indeks Kerentanan)</b><h2>{df["Kg"].mean():.2f}</h2></div>', unsafe_allow_html=True)
        
        st.write("")
        st.dataframe(df[['Titik', 'Longitude', 'Latitude', 'f0', 'A0', 'Kg', 'Tingkat Kerentanan', 'Karakteristik Tanah']], use_container_width=True)
        
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            st.write("**Histogram Distribusi Jumlah Titik berdasarkan Nilai Kg**")
            fig_hist = go.Figure(data=[go.Histogram(x=df["Kg"], marker_color='#0D9488')])
            fig_hist.update_layout(xaxis_title="Indeks Kerentanan Seismik (Kg)", yaxis_title="Jumlah Titik", height=320)
            st.plotly_chart(fig_hist, use_container_width=True)
        with g_col2:
            st.write("**Scatter Plot Hubungan f0 vs A0 terhadap Nilai Kg**")
            fig_scatter = go.Figure(data=go.Scatter(
                x=df["f0"], y=df["A0"], mode='markers', 
                marker=dict(size=df["Kg"]*2.0, color=df["Kg"], colorscale='jet', showscale=True)
            ))
            fig_scatter.update_layout(xaxis_title="Frekuensi Dominan f0 (Hz)", yaxis_title="Faktor Amplifikasi A0", height=320)
            st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# MENU 2: MIKROZONASI SPASIAL (4 OUTPUT PETA SATELIT TERINTEGRASI)
# ==========================================
elif menu == "Mikrozonasi Spasial":
    if df is None:
        st.warning("Silakan unggah file CSV data lapangan Anda terlebih dahulu pada panel di atas.")
    else:
        st.markdown('<div class="section-title">Output Peta Mikrozonasi Spasial (Overlay Kontur Surfer & Google Satellite)</div>', unsafe_allow_html=True)
        
        # Geometri Dasar & Batas Koordinat Lapangan
        center_lat, center_lon = df['Latitude'].mean(), df['Longitude'].mean()
        pad = 0.003
        min_lat, max_lat = df['Latitude'].min() - pad, df['Latitude'].max() + pad
        min_lon, max_lon = df['Longitude'].min() - pad, df['Longitude'].max() + pad
        bounds = [[min_lat, min_lon], [max_lat, max_lon]]
        
        # Proses Gridding Interpolasi Spasial IDW
        x, y = df['Longitude'].values, df['Latitude'].values
        x_line = np.linspace(min_lon, max_lon, 150)
        y_line = np.linspace(min_lat, max_lat, 150)
        xi, yi = np.meshgrid(x_line, y_line)
        
        zi_a0 = idw_interpolation(x, y, df['A0'].values, xi, yi)
        zi_f0 = idw_interpolation(x, y, df['f0'].values, xi, yi)
        zi_kg = idw_interpolation(x, y, df['Kg'].values, xi, yi)
        
        # Pembuatan Matriks Gambar Overlay Kontur bergaya Surfer
        img_a0 = create_contour_overlay(xi, yi, zi_a0, 'viridis')
        img_f0 = create_contour_overlay(xi, yi, zi_f0, 'plasma')
        img_kg = create_contour_overlay(xi, yi, zi_kg, 'jet')
        
        # FUNGSI GENERATOR UTAMA PETA SATELLITE + OVERLAY KONTUR + INTERACTIVE POPUP
        def generate_mikrozonasi_map(overlay_img=None):
            m = folium.Map(
                location=[center_lat, center_lon], zoom_start=15,
                tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                attr='Google Satellite Imagery'
            )
            
            # Pasang Gambar Kontur Berwarna Padat (Filled Contours) di atas Satelit jikalau parameter diisi
            if overlay_img is not None:
                folium.raster_layers.ImageOverlay(
                    image=overlay_img,
                    bounds=bounds,
                    opacity=0.6,  # Transparansi agar peta dasar satelit bumi tetap terlihat di bawah kontur
                    mercator_project=True
                ).add_to(m)
                
            # Pasang Marker Titik Pengukuran Lapangan dengan Fitur Interaktif Klik Laporan Detail
            for _, row in df.iterrows():
                popup_html = f"""
                <div style='font-family: Arial, sans-serif; font-size: 12px; width: 180px;'>
                    <h5 style='margin:0 0 5px 0; color:#1E3A8A; border-bottom:1px solid #CCC; padding-bottom:3px;'><b>Stasiun {row['Titik']}</b></h5>
                    <b>Latitude:</b> {row['Latitude']:.5f}<br>
                    <b>Longitude:</b> {row['Longitude']:.5f}<br>
                    <hr style='margin:5px 0;'>
                    <span style='color:#0D9488;'><b>f₀ (Frekuensi):</b> {row['f0']:.2f} Hz</span><br>
                    <span style='color:#EA580C;'><b>A₀ (Amplifikasi):</b> {row['A0']:.2f}</span><br>
                    <span style='background-color:#FEF2F2; padding:2px 4px; border-radius:3px; display:inline-block; margin-top:3px;'>
                        <b>K_g (Kerentanan):</b> <span style='color:#991B1B;'><b>{row['Kg']:.2f}</b></span>
                    </span>
                </div>
                """
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']], radius=7,
                    popup=folium.Popup(popup_html, max_width=250),
                    color='black', weight=1.5,
                    fill=True, fill_color=color_picker_kg(row['Tingkat Kerentanan']), fill_opacity=0.9
                ).add_to(m)
            return m

        # TAB LAYOUTING UNTUK MENAMPILKAN 4 OUTPUT SEKALIGUS SECARA RAPI
        tab1, tab2, tab3, tab4 = st.tabs([
            "📌 Peta 1: Peta Sebaran Stasiun", 
            "🟢 Peta 2: Kontur Amplifikasi (A₀)", 
            "🟣 Peta 3: Kontur Frekuensi Dominan (f₀)", 
            "🔴 Peta 4: Kontur Kerentanan Seismik (K_g)"
        ])
        
        with tab1:
            st.markdown("#### Peta 1: Lokasi Titik Pengukuran Geofisika Lapangan")
            st.caption("💡 Klik lingkaran pin untuk memunculkan panel pop-up berisi nomor stasiun dan nilai indeks kerentanan ($K_g$) spesifik.")
            map_stasiun = generate_mikrozonasi_map(overlay_img=None)
            st_folium(map_stasiun, width=1100, height=450, key="map_stasiun")
            
        with tab2:
            st.markdown("#### Peta 2: Kuantifikasi Faktor Amplifikasi Lapisan ($A_0$)")
            st.caption("Model interpolasi warna padat (Filled Contour lines) menggunakan skala standar Matplotlib-Surfer `Viridis` Layer.")
            map_a0 = generate_mikrozonasi_map(overlay_img=img_a0)
            st_folium(map_a0, width=1100, height=450, key="map_a0")
            
        with tab3:
            st.markdown("#### Peta 3: Distribusi Nilai Frekuensi Alami Batuan Dasar ($f_0$)")
            st.caption("Model zonasi batuan keras hingga sedimen lunak bersumber dari hasil gridding penampang warna `Plasma` Layer.")
            map_f0 = generate_mikrozonasi_map(overlay_img=img_f0)
            st_folium(map_f0, width=1100, height=450, key="map_f0")
            
        with tab4:
            st.markdown("#### Peta 4: Peta Zonasi Mikro Indeks Kerentanan Seismik ($K_g$)")
            st.caption("Zona kritis anomali tinggi ditandai dengan warna merah pekat (`Jet Colormap`). Sangat vital sebagai acuan tata ruang konstruksi.")
            map_kg = generate_mikrozonasi_map(overlay_img=img_kg)
            st_folium(map_kg, width=1100, height=450, key="map_kg")

# ==========================================
# MENU 3: ANALISIS RESONANSI
# ==========================================
elif menu == "Analisis Resonansi":
    if df is None:
        st.warning("Silakan unggah file CSV data lapangan Anda terlebih dahulu pada panel di atas.")
    else:
        st.markdown('<div class="section-title">Analisis Risiko Resonansi Struktur terhadap Bangunan Sekitar</div>', unsafe_allow_html=True)
        
        num_floors = st.number_input("Masukkan Jumlah Lantai Bangunan Target yang Disimulasikan (N):", min_value=1, max_value=40, value=3)
        fb = 10.0 / num_floors
        
        st.markdown(f'<div class="metric-box"><b>Karakteristik Frekuensi Alami Bangunan Rencana (fb):</b> <h2>{fb:.2f} Hz (Untuk Spesifikasi Bangunan Berlantai {num_floors})</h2></div>', unsafe_allow_html=True)
        
        def evaluate_resonance_risk(f0, fb):
            diff = abs(f0 - fb)
            if diff <= 0.5: 
                return "Risiko Tinggi (Bahaya Resonansi)"
            elif 0.5 < diff <= 1.5: 
                return "Risiko Sedang (Waspada)"
            return "Risiko Rendah (Aman)"
            
        df_res = df[['Titik', 'Longitude', 'Latitude', 'f0', 'A0']].copy()
        df_res['fb (Freq Bangunan)'] = round(fb, 2)
        df_res['Selisih |f0 - fb| (Hz)'] = round(abs(df_res['f0'] - fb), 2)
        df_res['Kondisi Kerentanan Bangunan'] = df_res.apply(lambda row: evaluate_resonance_risk(row['f0'], fb), axis=1)
        
        st.write("")
        st.markdown("**Tabel Hasil Komparasi Parameter Frekuensi Dominan Tanah vs Bangunan Sekitar:**")
        
        def color_rows(row):
            status = row['Kondisi Kerentanan Bangunan']
            if status == "Risiko Tinggi (Bahaya Resonansi)":
                return ['background-color: #FEE2E2; color: #991B1B'] * len(row)
            elif status == "Risiko Sedang (Waspada)":
                return ['background-color: #FEF3C7; color: #92400E'] * len(row)
            return ['background-color: #DCFCE7; color: #166534'] * len(row)
            
        styled_df = df_res.style.apply(color_rows, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        st.markdown("""
        <div class="ref-box">
        <b>Catatan Interpretasi Risiko Resonansi Struktur:</b><br>
        • <b>Risiko Tinggi (Merah):</b> Selisih frekuensi alami tanah ($f_0$) dan bangunan ($f_b$) sangat dekat ($\leq 0.5$ Hz). Bangunan rentan mengalami guncangan ganda ekstrem jika terjadi gempa bumi karena gelombang saling menguatkan.<br>
        • <b>Risiko Sedang (Kuning):</b> Selisih frekuensi berada di rentang $0.5$ hingga $1.5$ Hz. Direkomendasikan melakukan perkuatan struktural atau rekayasa kekakuan kolom fondasi utama.<br>
        • <b>Risiko Rendah (Hijau):</b> Selisih frekuensi $> 1.5$ Hz. Struktur aman dari bahaya amplifikasi kerusakan akibat getaran resonansi lapisan tanah lokal.
        </div>
        """, unsafe_allow_html=True)
