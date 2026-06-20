import streamlit as st
import pandas as pd
import numpy as np
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
    page_title="Mikrozonasi Seismik HVSR",
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
    .ref-box { background-color: #EFF6FF; padding: 15px; border-radius: 8px; border-left: 5px solid #3B82F6; margin-top: 15px; font-size: 13px; color: #1E3A8A; }
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

def classify_amplification(a0):
    if a0 < 3: return "Low"
    elif 3 <= a0 < 6: return "Moderate"
    elif 6 <= a0 < 9: return "High"
    else: return "Very High"

def classify_soil_kanai(f0):
    if 6.7 <= f0 <= 20: 
        return "Klas I (Tanah Keras / Batuan Padat)"
    elif 4 <= f0 < 6.7: 
        return "Klas II (Tanah Sedang / Aluvial Dangkal)"
    elif 2.5 <= f0 < 4: 
        return "Klas III (Tanah Sedang-Lunak)"
    elif f0 < 2.5: 
        return "Klas IV (Tanah Lunak / Sedimen Tebal)"
    else: 
        return "Di luar Rentang Standar Kanai (>20 Hz)"

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

def create_surfer_contour_overlay(xi, yi, zi, cmap_name, levels=15):
    fig, ax = plt.subplots(figsize=(10, 10), dpi=200, frameon=False)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off') 
    
    ax.contourf(xi, yi, zi, levels=levels, cmap=cmap_name)
    ax.contour(xi, yi, zi, levels=levels, colors='black', linewidths=0.4, alpha=0.6)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    img_data = plt.imread(buf)
    
    plt.close(fig)
    buf.close()
    return img_data

# Inisialisasi Session State Data
if 'df_data' not in st.session_state:
    st.session_state.df_data = None

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #1E3A8A; margin-top: 20px;'>🌋 GEOFISIKA</h2>", unsafe_allow_html=True)
    st.markdown("---")
    menu = st.radio("Pilih Menu Dashboard:", ["Analisis Kerentanan", "Mikrozonasi Spasial", "Analisis Resonansi"])
    st.markdown("---")

# ==========================================
# 4. HEADER UTAMA & UPLOAD DATA
# ==========================================
st.markdown('<div class="main-title">Aplikasi Mikrozonasi Seismik HVSR</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Dashboard Analisis Karakteristik Dinamik Tanah & Kerentanan Seismik</div>', unsafe_allow_html=True)

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
            df['Klasifikasi Klas Tanah (Kanai)'] = df['f0'].apply(classify_soil_kanai)
            df['Klasifikasi Amplifikasi Situs'] = df['A0'].apply(classify_amplification)
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
        st.dataframe(df
