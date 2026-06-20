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
    st.markdown("<h2 style='text-align: center; color: #1E3A8A; margin-top
