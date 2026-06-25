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
# 4. DATA HANDLING ENTRY & CUSTOM MAPPER
# ==========================================
st.markdown('<div class="main-title">Aplikasi Mikrozonasi Seismik HVSR</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Dashboard Analisis Karakteristik Dinamik Tanah & Kerentanan Seismik</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Unggah File CSV Pengukuran Mikrotremor:", type=["csv"])

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file, sep=None, engine='python')
        df_raw.columns = df_raw.columns.str.strip() 
        
        st.markdown('<div class="section-title">⚙️ Konfigurasi Penyesuaian Format Kolom (CSV Mapper)</div>', unsafe_allow_html=True)
        st.info("Format kolom file Anda tidak langsung cocok? Silakan sesuaikan pilihan drop-down di bawah ini agar sesuai dengan format aplikasi.")
        
        def auto_guess(options, keywords):
            for opt in options:
                if any(k in opt.lower() for k in keywords):
                    return options.index(opt)
            return 0

        columns_list = list(df_raw.columns)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            mapped_titik = st.selectbox("1. Kolom Identitas / Nama Titik:", columns_list, index=auto_guess(columns_list, ['titik', 'nama', 'station', 'id']))
        with col2:
            mapped_lon = st.selectbox("2. Kolom Bujur / Longitude (X):", columns_list, index=auto_guess(columns_list, ['long', 'lng', 'longitude', 'x']))
        with col3:
            mapped_lat = st.selectbox("3. Kolom Lintang / Latitude (Y):", columns_list, index=auto_guess(columns_list, ['lat', 'latitude', 'y']))
            
        col4, col5, col6 = st.columns(3)
        with col4:
            mapped_f0 = st.selectbox("4. Kolom Frekuensi Dominan (f0):", columns_list, index=auto_guess(columns_list, ['f0', 'fo', 'frekuensi', 'frequency']))
        with col5:
            mapped_a0 = st.selectbox("5. Kolom Faktor Amplifikasi (A0):", columns_list, index=auto_guess(columns_list, ['a0', 'ao', 'amplifikasi', 'ampli']))
        with col6:
            st.markdown("<br>", unsafe_allow_html=True)
            proses_btn = st.button("🔄 Terapkan & Proses Data", use_container_width=True, type="primary")

        if proses_btn:
            df_mapped = df_raw.rename(columns={
                mapped_titik: 'Titik',
                mapped_lon: 'Longitude',
                mapped_lat: 'Latitude',
                mapped_f0: 'f0',
                mapped_a0: 'A0'
            })
            
            required_cols = ['Titik', 'Longitude', 'Latitude', 'f0', 'A0']
            df_final = df_mapped[required_cols].copy()
            
            for c in ['f0', 'A0', 'Latitude', 'Longitude']:
                df_final[c] = pd.to_numeric(df_final[c], errors='coerce')
                
            df_final = df_final.dropna(subset=['f0', 'A0', 'Latitude', 'Longitude'])
            
            df_final['Kg'] = calculate_kg(df_final['A0'], df_final['f0'])
            df_final['Tingkat Kerentanan'] = df_final['Kg'].apply(classify_kg)
            df_final['Klasifikasi Klas Tanah (Kanai)'] = df_final['f0'].apply(classify_soil_kanai)
            df_final['Klasifikasi Amplifikasi Situs'] = df_final['A0'].apply(classify_amplification)
            
            st.session_state.df_data = df_final
            st.success("🎉 Data berhasil disesuaikan, dihitung, dan dimuat ke dalam visualisasi dashboard!")
            
    except Exception as e:
        st.error(f"Gagal membaca struktur file CSV Anda: {e}")

df = st.session_state.df_data

# ==========================================
# MENU 1: ANALISIS KERENTANAN
# ==========================================
if menu == "Analisis Kerentanan":
    if df is None:
        st.warning("Silakan unggah file CSV dan klik tombol 'Terapkan & Proses Data' di atas.")
    else:
        st.markdown('<div class="section-title">Ringkasan Statistik Lapisan</div>', unsafe_allow_html=True)
        f0_avg, f0_min, f0_max = df["f0"].mean(), df["f0"].min(), df["f0"].max()
        a0_avg, a0_min, a0_max = df["A0"].mean(), df["A0"].min(), df["A0"].max()
        kg_avg, kg_min, kg_max = df["Kg"].mean(), df["Kg"].min(), df["Kg"].max()
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="card-f0">
                <div class="card-title">🔵 Rata-rata f0</div>
                <div class="card-value">{f0_avg:.2f} Hz</div>
                <div class="card-sub">Min: {f0_min:.2f} Hz &nbsp;|&nbsp; Max: {f0_max:.2f} Hz</div>
            </div>
            <div class="card-a0">
                <div class="card-title">🟠 Rata-rata A0</div>
                <div class="card-value">{a0_avg:.2f}</div>
                <div class="card-sub">Min: {a0_min:.2f} &nbsp;|&nbsp; Max: {a0_max:.2f}</div>
            </div>
            <div class="card-kg">
                <div class="card-title">🟢 Rata-rata Kg</div>
                <div class="card-value">{kg_avg:.2f}</div>
                <div class="card-sub">Min: {kg_min:.2f} &nbsp;|&nbsp; Max: {kg_max:.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-title">Tabel Hasil Analisis Titik Pengukuran</div>', unsafe_allow_html=True)
        df_view = df.copy()
        df_view.insert(0, 'No.', range(1, len(df_view) + 1))
        
        def style_hvsr_table(val):
            if "Klas I" in str(val): return 'background-color: #DCFCE7; color: #15803D; font-weight: bold;'
            elif "Klas II" in str(val): return 'background-color: #E0F2FE; color: #0369A1; font-weight: bold;'
            elif "Klas III" in str(val): return 'background-color: #FEF3C7; color: #B45309; font-weight: bold;'
            elif "Klas IV" in str(val): return 'background-color: #FEE2E2; color: #B91C1C; font-weight: bold;'
            elif val in ["Low", "Rendah"]: return 'background-color: #DCFCE7; color: #166534; text-align: center;'
            elif val in ["Moderate", "Menengah"]: return 'background-color: #FEF3C7; color: #92400E; text-align: center;'
            elif val in ["High", "Tinggi", "Very High"]: return 'background-color: #FEE2E2; color: #991B1B; text-align: center;'
            return ''

        styled_table = df_view[[
            'No.', 'Titik', 'Longitude', 'Latitude', 'f0', 
            'Klasifikasi Klas Tanah (Kanai)', 'A0', 'Klasifikasi Amplifikasi Situs', 
            'Kg', 'Tingkat Kerentanan'
        ]].style.map(style_hvsr_table, subset=[
            'Klasifikasi Klas Tanah (Kanai)', 'Klasifikasi Amplifikasi Situs', 'Tingkat Kerentanan'
        ])
        
        st.dataframe(styled_table, use_container_width=True)

# ==========================================
# MENU 2: MIKROZONASI SPASIAL
# ==========================================
elif menu == "Mikrozonasi Spasial":
    if df is None:
        st.warning("Silakan unggah file CSV dan klik tombol 'Terapkan & Proses Data' di atas.")
    else:
        st.markdown('<div class="section-title">Output Peta Mikrozonasi Spasial (Google Satellite)</div>', unsafe_allow_html=True)
        
        center_lat, center_lon = df['Latitude'].mean(), df['Longitude'].mean()
        pad = 0.002  
        min_lat, max_lat = df['Latitude'].min() - pad, df['Latitude'].max() + pad
        min_lon, max_lon = df['Longitude'].min() - pad, df['Longitude'].max() + pad
        bounds = [[min_lat, min_lon], [max_lat, max_lon]]
        
        x, y = df['Longitude'].values, df['Latitude'].values
        x_line =
