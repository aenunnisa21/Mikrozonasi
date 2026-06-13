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
                
                df['Kg'] = calculate_kg(df['A0'], df
