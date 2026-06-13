import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from scipy.spatial import distance_matrix

# ==========================================
# KONFIGURASI HALAMAN UTAMA
# ==========================================
st.set_page_config(
    page_title="Mikrozonasi Seismik HVSR - UIN Sunan Kalijaga",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk layout dashboard Geofisika
st.markdown("""
    <style>
    .main-title { font-size: 32px; font-weight: bold; color: #1E3A8A; margin-bottom: 20px; }
    .section-title { font-size: 24px; font-weight: bold; color: #0D9488; margin-top: 25px; margin-bottom: 15px; border-bottom: 2px solid #0D9488; padding-bottom: 5px; }
    .metric-box { background-color: #F3F4F6; padding: 15px; border-radius: 10px; border-left: 5px solid #1E3A8A; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    .ref-box { background-color: #EFF6FF; padding: 15px; border-radius: 8px; border-left: 5px solid #3B82F6; margin-top: 15px; font-size: 13px; color: #1E3A8A; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# FUNGSI GEOFISIKA & INTERPOLASI
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

if 'df_data' not in st.session_state:
    st.session_state.df_data = None

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    try:
        st.image("logo_uin.png", use_container_width=True)
    except:
        st.markdown("<h2 style='text-align: center; color: #1E3A8A;'>🕌 UIN SUKA</h2>", unsafe_allow_html=True)
            
    st.markdown("<h3 style='text-align: center; margin-top: 10px; color: #111827; font-size: 18px;'>GEOFISIKA UIN SUKA</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 13px; color: #6B7280; margin-top: -10px;'>Analisis Mikrotremor HVSR</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    menu = st.radio(
        "Pilih Menu Navigasi:",
        ["Home", "Upload Data", "Analisis Kerentanan", "Mikrozonasi", "Analisis Resonansi"]
    )
    st.markdown("---")
    st.caption("UAS Seismologi - © 2026")

# ==========================================
# MENU 1: HOME
# ==========================================
if menu == "Home":
    st.markdown('<div class="main-title">Aplikasi Mikrozonasi Seismik Berbasis Mikrotremor (HVSR)</div>', unsafe_allow_html=True)
    st.subheader("Program Studi Geofisika - UIN Sunan Kalijaga Yogyakarta")
    st.markdown("""
    ### Deskripsi Aplikasi
    Aplikasi ini dirancang untuk melakukan standarisasi pengolahan lanjut data hasil *Horizontal-to-Vertical Spectral Ratio* (HVSR) mikrotremor. 
    Melalui integrasi parameter frekuensi dominan ($f_0$) dan amplifikasi ($A_0$), sistem melakukan pemetaan bahaya lokal (*local site effect*) guna mendukung mitigasi bencana gempa bumi regional.
    """)

# ==========================================
# MENU 2: UPLOAD DATA
# ==========================================
elif menu == "Upload Data":
    st.markdown('<div class="main-title">Upload Data Mikrotremor</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
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
                st.success("Data berhasil diunggah!")
                st.dataframe(df, use_container_width=True)
            else:
                st.error("Format salah. Pastikan kolom berisi: TITIK, LONGITUDE, LATITUDE, F0, A0")
        except Exception as e:
            st.error(f"Error: {e}")
    elif st.session_state.df_data is not None:
        st.dataframe(st.session_state.df_data, use_container_width=True)

# ==========================================
# MENU 3: ANALISIS KERENTANAN
# ==========================================
elif menu == "Analisis Kerentanan":
    st.markdown('<div class="main-title">Analisis Kerentanan Seismik & Statistik Detail</div>', unsafe_allow_html=True)
    df = st.session_state.df_data
    if df is None:
        st.warning("Silakan upload data CSV terlebih dahulu di menu Upload Data.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1: 
            st.markdown(f'<div class="metric-box"><b>Rata-rata f0 (Frekuensi Dominan)</b><h2>{df["f0"].mean():.2f} Hz</h2></div>', unsafe_allow_html=True)
        with col2: 
            st.markdown(f'<div class="metric-box"><b>Rata-rata A0 (Amplifikasi)</b><h2>{df["A0"].mean():.2f}</h2></div>', unsafe_allow_html=True)
        with col3: 
            st.markdown(f'<div class="metric-box"><b>Rata-rata Kg (Indeks Kerentanan)</b><h2>{df["Kg"].mean():.2f}</h2></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-title">Tabel Ringkasan Statistik Parameter HVSR</div>', unsafe_allow_html=True)
        desc_df = df[['f0', 'A0', 'Kg']].describe().T
        desc_df = desc_df.rename(columns={'mean': 'Rata-rata', 'std': 'Deviasi Standar', 'min': 'Nilai Min', 'max': 'Nilai Maks', '50%': 'Median'})
        st.dataframe(desc_df[['Rata-rata', 'Deviasi Standar', 'Nilai Min', 'Median', 'Nilai Maks']], use_container_width=True)
        
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            st.write("**Histogram Distribusi Jumlah Titik berdasarkan Nilai Kg**")
            fig_hist = go.Figure(data=[go.Histogram(x=df["Kg"], marker_color='#0D9488')])
            fig_hist.update_layout(xaxis_title="Indeks Kerentanan Seismik (Kg)", yaxis_title="Jumlah Titik", height=350)
            st.plotly_chart(fig_hist, use_container_width=True)
        with g_col2:
            st.write("**Scatter Plot Hubungan f0 vs A0 terhadap Nilai Kg**")
            fig_scatter = go.Figure(data=go.Scatter(
                x=df["f0"], y=df["A0"], mode='markers', 
                marker=dict(size=df["Kg"]*1.5, color=df["Kg"], colorscale='Jet', showscale=True)
            ))
            fig_scatter.update_layout(xaxis_title="Frekuensi Dominan f0 (Hz)", yaxis_title="Faktor Amplifikasi A0", height=350)
            st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# MENU 4: MIKROZONASI (SURFER SOFTWARE ENGINE REPLICA)
# ==========================================
elif menu == "Mikrozonasi":
    st.markdown('<div class="main-title">Peta Kerentanan Seismik & Model Spasial 3D</div>', unsafe_allow_html=True)
    df = st.session_state.df_data
    if df is None:
        st.warning("Silakan upload data CSV terlebih dahulu di menu Upload Data.")
    else:
        # 1. PETA GIS MAPBOX (MENAMPILKAN PERMUKAAN BUMI NYATA / RELEIF TERRAIN)
        st.markdown('<div class="section-title">Peta 1: GIS Overlay Parameter Seismik pada Basemap Terrain Bumi</div>', unsafe_allow_html=True)
        center_lat, center_lon = df['Latitude'].mean(), df['Longitude'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon], zoom_start=14,
            tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
            attr='&copy; OpenTopoMap'
        )
        for _, row in df.iterrows():
            popup_html = f"<b>Stasiun {row['Titik']}</b><br>f₀: {row['f0']:.2f} Hz<br>A₀: {row['A0']:.2f}<br>K_g: {row['Kg']:.2f}"
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']], radius=9,
                popup=folium.Popup(popup_html, max_width=200),
                color=color_picker_kg(row['Tingkat Kerentanan']), fill=True, fill_opacity=0.85
            ).add_to(m)
        st_folium(m, width=1100, height=350)
        
        # 2. PEMBUATAN REPLIKA MODEL 3D VOLCANO/HILLSHADE RELEIF SURFACE Khas SURFER
        x, y = df['Longitude'].values, df['Latitude'].values
        x_line = np.linspace(x.min() - 0.002, x.max() + 0.002, 70)
        y_line = np.linspace(y.min() - 0.002, y.max() + 0.002, 70)
        xi, yi = np.meshgrid(x_line, y_line)
        
        # Generator Topografi Buatan Tinggi-Rendah (Sumbu Z) agar membentuk morfologi gunung/bukit bertekstur tajam
        r = np.sqrt((xi - center_lon)**2 + (yi - center_lat)**2) * 1200
        z_mountain = 1000 * np.exp(-r/6) + 140 * np.sin(xi*1600) * np.cos(yi*1600)
        
        # Interpolasi nilai HVSR untuk disuntikkan sebagai warna penutup lereng
        zi_a0 = idw_interpolation(x, y, df['A0'].values, xi, yi)
        zi_f0 = idw_interpolation(x, y, df['f0'].values, xi, yi)
        zi_kg = idw_interpolation(x, y, df['Kg'].values, xi, yi)
        
        st.markdown('<div class="section-title">Peta 2: 3D Surface Model Parameter Seismik (Surfer Style)</div>', unsafe_allow_html=True)
        col_3a, col_3b, col_3c = st.columns(3)
