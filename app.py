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

# Custom CSS untuk meniru persis gaya UI di image_24c2e0.jpg
st.markdown("""
    <style>
    .main-title { font-size: 30px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
    .sub-title { font-size: 16px; color: #4B5563; margin-bottom: 25px; }
    .section-title { font-size: 20px; font-weight: bold; color: #0D9488; margin-top: 20px; margin-bottom: 15px; border-bottom: 2px solid #0D9488; padding-bottom: 5px; }
    
    /* Style Kartu Metrik Ringkasan Statistik */
    .metric-container { display: flex; gap: 15px; margin-bottom: 20px; }
    .card-f0 { background-color: #F0F6FF; border-left: 6px solid #2563EB; border-radius: 12px; padding: 15px; flex: 1; box-shadow: 0 2px 4px rgba(0,0,0,0.04); }
    .card-a0 { background-color: #FFF7ED; border-left: 6px solid #EA580C; border-radius: 12px; padding: 15px; flex: 1; box-shadow: 0 2px 4px rgba(0,0,0,0.04); }
    .card-kg { background-color: #F0FDF4; border-left: 6px solid #16A34A; border-radius: 12px; padding: 15px; flex: 1; box-shadow: 0 2px 4px rgba(0,0,0,0.04); }
    .card-title { font-size: 13px; font-weight: bold; color: #4B5563; margin-bottom: 2px; }
    .card-value { font-size: 28px; font-weight: bold; color: #1F2937; margin: 0; }
    .card-sub { font-size: 11px; color: #6B7280; margin-top: 4px; }
    
    /* Keterangan Parameter Sidebar Box */
    .sidebar-desc-box { background-color: #EFF6FF; border-radius: 10px; padding: 12px; border: 1px solid #BFDBFE; margin-top: 30px; }
    .sidebar-desc-title { font-size: 13px; font-weight: bold; color: #1E40AF; margin-bottom: 8px; }
    .sidebar-desc-item { font-size: 11px; color: #1E3A8A; margin-bottom: 6px; line-height: 1.3; }
    
    /* Legenda bawah */
    .legend-box { background-color: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 8px; padding: 15px; margin-top: 15px; }
    .legend-col-title { font-size: 12px; font-weight: bold; color: #374151; margin-bottom: 6px; border-bottom: 1px solid #E5E7EB; padding-bottom: 3px; }
    .legend-item { font-size: 11px; color:
