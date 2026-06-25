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

# Custom CSS untuk menyelaraskan gaya visual dashboard
st.markdown("""
    <style>
    .main-title { font-size: 30px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
    .sub-title { font-size: 16px; color: #4B5563; margin-bottom: 25px; }
    .section-title { font-size: 20px; font-weight: bold; color: #0D9488; margin-top: 20px; margin-bottom: 15px; border-bottom: 2px solid #0D9488; padding-bottom: 5px; }
    
    /* Gaya Grid Kartu Metrik Ringkasan */
    .metric-container { display: flex; gap: 15px; margin-bottom: 20px; }
    .card-f0 { background-color: #F0F6FF; border-left: 6px solid #2563EB; border-radius: 12px; padding: 15px; flex: 1; box-shadow: 0 2px 4px rgba(0,0,0,0.04); }
    .card-a0 { background-color: #FFF7ED; border-left: 6px solid #EA580C; border-radius: 12px; padding: 15px; flex: 1; box-shadow: 0 2px 4px rgba(0,0,0,0.04); }
    .card-kg { background-color: #F0FDF4; border-left: 6px solid #16A34A; border-radius: 12px; padding: 15px; flex: 1; box-shadow: 0 2px 4px rgba(0,0,0,0.04); }
    
    /* Gaya Kartu Bahaya Resonansi (Menu 3) */
    .card-danger-high { background-color: #FEE2E2; border-left: 6px solid #DC2626; border-radius: 12px; padding: 15px; flex: 1; box-shadow: 0 2px 4px rgba(0,0,0,0.04); }
    .card-danger-mid { background-color: #FEF3C7; border-left: 6px solid #D97706; border-radius: 12px; padding: 15px; flex: 1; box-shadow: 0 2px 4px rgba(0,0,0,0.04); }
    .card-danger-low { background-color: #DCFCE7; border-left: 6px solid #16A34A; border-radius: 12px; padding: 15px; flex: 1; box-shadow: 0 2px 4px rgba(0,0,0,0.04); }
    
    .card-title { font-size: 13px; font-weight: bold; color: #4B5563; margin-bottom: 2px; }
    .card-value { font-size: 26px; font-weight: bold; color: #1F2937; margin: 0; }
    .card-value-high { font-size: 22px; font-weight: bold; color: #DC2626; margin: 0; }
    .card-value-mid { font-size: 22px; font-weight: bold; color: #D97706; margin: 0; }
    .card-value-low { font-size: 22px; font-weight: bold; color: #16A34A; margin: 0; }
    .card-sub { font-size: 11px; color: #6B7280; margin-top: 4px; }
    
    /* Panel Informasi Tambahan */
    .sidebar-desc-box { background-color: #EFF6FF; border-radius: 10px; padding: 12px; border: 1px solid #BFDBFE; margin-top: 30px; }
    .sidebar-desc-title { font-size: 13px; font-weight: bold; color: #1E40AF; margin-bottom: 8px; }
    .sidebar-desc-item { font-size: 11px; color: #1E3A8A; margin-bottom: 6px; line-height: 1.3; }
    .legend-box { background-color: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 8px; padding: 15px; margin-top: 15px; }
    .legend-col-title { font-size: 12px; font-weight: bold; color: #374151; margin-bottom: 6px; border-bottom: 1px solid #E5E7EB; padding-bottom: 3px; }
    .legend-item { font-size: 11px; color: #4B5563; margin-bottom: 3px; display: flex; align-items: center; }
    .dot { height: 8px; width: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; }
    .ref-box { background-color: #F8FAFC; border-left: 4px solid #64748B; padding: 12px; border-radius: 6px; font-size: 12px; color: #334155; line-height: 1.5; margin-top: 15px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. FUNGSI UTAMA (MATEMATIKA & PROSES)
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
    if 6.7 <= f0 <= 20: return "Klas I (Tanah Keras)"
    elif 4 <= f0 < 6.7: return "Klas II (Tanah Sedang)"
    elif 2.5 <= f0 < 4: return "Klas III (Tanah Sedang-Lunak)"
    elif f0 < 2.5: return "Klas IV (Tanah Lunak)"
    else: return "Tanah Keras (>20 Hz)"

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

if 'df_data' not in st.session_state:
    st.session_state.df_data = None

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #1E3A8A; margin-top: 15px;'>🌋 GEOFISIKA</h2>", unsafe_allow_html=True)
    st.markdown("---")
    menu = st.radio("Pilih Menu Dashboard:", ["Analisis Kerentanan", "Mikrozonasi Spasial", "Analisis Resonansi"])
    st.markdown("---")
    
    st.markdown("""
    <div class="sidebar-desc-box">
        <div class="sidebar-desc-title">🌐 Keterangan Parameter</div>
        <div class="sidebar-desc-item"><b>📈 f0 (Hz):</b> Frekuensi dominan tanah hasil analisis perbandingan HVSR.</div>
        <div class="sidebar-desc-item"><b>📉 A0:</b> Amplifikasi maksimum atau faktor perbesaran gelombang situs.</div>
        <div class="sidebar-desc-item"><b>🛡️ Kg:</b> Indeks kerentanan seismik lapisan tanah makro (Kg = A₀² / f₀).</div>
        <div class="sidebar-desc-item"><b>🧱 Klas Tanah:</b> Klasifikasi jenis batuan berdasarkan periode getar Kanai (1957).</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 4. DATA HANDLING ENTRY
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
# MENU 1: ANALISIS KERENTANAN
# ==========================================
if menu == "Analisis Kerentanan":
    if df is None:
        st.warning("Silakan unggah data lapangan CSV terlebih dahulu.")
    else:
        st.markdown('<div class="section-title">Ringkasan Statistik Lapisan</div>', unsafe_allow_html=True)
        f0_avg, f0_min, f0_max = df["f0"].mean(), df["f0"].min(), df
