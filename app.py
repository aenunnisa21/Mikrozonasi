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

st.markdown("""
    <style>
    .main-title { font-size: 32px; font-weight: bold; color: #1E3A8A; margin-bottom: 20px; }
    .section-title { font-size: 24px; font-weight: bold; color: #0D9488; margin-top: 25px; margin-bottom: 15px; border-bottom: 2px solid #0D9488; padding-bottom: 5px; }
    .metric-box { background-color: #F3F4F6; padding: 15px; border-radius: 10px; border-left: 5px solid #1E3A8A; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    .ref-box { background-color: #EFF6FF; padding: 15px; border-radius: 8px; border-left: 5px solid #3B82F6; margin-top: 15px; font-size: 13px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# GEOPHYSICAL FUNCTIONS & REFERENCES
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
# MENU 4: MIKROZONASI (3D SOLID BLOCK MODEL ALA SOFTWARE SURFER)
# ==========================================
elif menu == "Mikrozonasi":
    st.markdown('<div class="main-title">Peta Kerentanan Seismik & Model Spasial 3D</div>', unsafe_allow_html=True)
    df = st.session_state.df_data
    if df is None:
        st.warning("Silakan upload data CSV terlebih dahulu di menu Upload Data.")
    else:
        # PROSES INTERPOLASI IDW GRID (RESOLUSI RAPAT UNTUK SURFER STYLE)
        x, y = df['Longitude'].values, df['Latitude'].values
        x_line = np.linspace(x.min() - 0.002, x.max() + 0.002, 60)
        y_line = np.linspace(y.min() - 0.002, y.max() + 0.002, 60)
        xi, yi = np.meshgrid(x_line, y_line)
        
        zi_a0 = idw_interpolation(x, y, df['A0'].values, xi, yi)
        zi_f0 = idw_interpolation(x, y, df['f0'].values, xi, yi)
        zi_kg = idw_interpolation(x, y, df['Kg'].values, xi, yi)
        
        st.markdown('<div class="section-title">3D Solid Block Model Parameter Seismik (Surfer Software Style)</div>', unsafe_allow_html=True)
        st.write("Model di bawah ini dikonstruksi menggunakan proyeksi dinding solid bawah dan kontur lantai dasar agar identik dengan output cetakan perangkat lunak Surfer.")
        
        col_3a, col_3b, col_3c = st.columns(3)
        
        # PENGATURAN TEKSTUR & PENCAHAYAAN MATTE ALA SURFER
        surfer_lighting = dict(
            ambient=0.7,      # Mengurangi bayangan gelap ekstrem
            diffuse=0.8,      # Menyebarkan warna merata di lereng
            fresnel=0.1,      # Menghilangkan pantulan kilap plastik
            specular=0.1,     # Permukaan doff/matte
            roughness=0.9
        )
        
        # FUNGSI UNTUK MEMBUAT BLOK 3D DENGAN PROYEKSI LANTAI & DINDING
        def create_surfer_block(zi_data, title_name, colorscale_name, z_min, z_max):
            fig = go.Figure(data=[go.Surface(
                z=zi_data, x=x_line, y=y_line,
                colorscale=colorscale_name,
                lighting=surfer_lighting,
                showscale=True,
                colorbar=dict(
                    title=title_name,
                    thickness=15,
                    len=0.7,
                    titleside="top"
                ),
                # 1. Membuat Garis Kontur di Atas Permukaan Bukit
                contours_z=dict(
                    show=True,
                    usecolormap=False,
                    highlightcolor="black",
                    project_z=True,   # <-- INI MEMBUAT PROYEKSI KONTUR DI LANTAI DASAR
                    start=float(np.min(zi_data)),
                    end=float(np.max(zi_data)),
                    size=float((np.max(zi_data) - np.min(zi_data)) / 12),
                    color="rgba(0, 0, 0, 0.4)",
                    width=1.5
                )
            )])
            
            # 2. Rekayasa Tampilan Dinding Kotak Padat (Solid Box Base)
            z_floor = float(np.min(zi_data) - (np.max(zi_data) - np.min(zi_data))*0.4)
            
            fig.update_layout(
                scene=dict(
                    xaxis=dict(title='Longitude', gridcolor='rgba(0,0,0,0.1)', backgroundcolor="white", showbackground=True),
                    yaxis=dict(title='Latitude', gridcolor='rgba(0,0,0,0.1)', backgroundcolor="white", showbackground=True),
                    zaxis=dict(
                        title=title_name, 
                        range=[z_floor, float(np.max(zi_data) * 1.1)],
                        gridcolor='rgba(0,0,0,0.1)',
                        backgroundcolor="rgba(230,230,230,0.5)", # Sisi samping abu-abu solid
                        showbackground=True
                    ),
                    aspectratio=dict(x=1, y=1, z=0.5),
                    camera=dict(eye=dict(x=1.3, y=-1.3, z=0.9)) # Sudut pandang isometrik khas Surfer
                ),
                margin=dict(l=0, r=0, b=0, t=30),
                height=480
            )
            return fig

        # Render 3 Peta Berjajar secara Otomatis
        with col_3a:
            st.markdown("<h4 style='text-align: center; color: #1E3A8A; font-size: 16px;'>3A. 3D Site Amplification (Ao)</h4>", unsafe_allow_html=True)
            fig_a0 = create_surfer_block(zi_a0, "Ao", "Viridis", df['A0'].min(), df['A0'].max())
            st.plotly_chart(fig_a0, use_container_width=True)
            
        with col_3b:
            st.markdown("<h4 style='text-align: center; color: #1E3A8A; font-size: 16px;'>3B. 3D Dominant Frequency (f0)</h4>", unsafe_allow_html=True)
            fig_f0 = create_surfer_block(zi_f0, "f0 (Hz)", "Plasma", df['f0'].min(), df['f0'].max())
            st.plotly_chart(fig_f0, use_container_width=True)
            
        with col_3c:
            st.markdown("<h4 style='text-align: center; color: #1E3A8A; font-size: 16px;'>3C. 3D Seismic Vulnerability (Kg)</h4>", unsafe_allow_html=True)
            fig_kg = create_surfer_block(zi_kg, "Kg", "Jet", df['Kg'].min(), df['Kg'].max())
            st.plotly_chart(fig_kg, use_container_width=True)

        st.markdown("""
        <div class="ref-box">
        <b>Pedoman Klasifikasi Lapisan Tanah (Klas Kanai, 1983 & Nakamura, 1997):</b><br>
        • <b>Anomali Bukit Tinggi / Warna Merah (Kg):</b> Merupakan zona rawan deformasi tinggi akibat akumulasi lapisan sedimen lunak lokal.<br>
        • <b>Proyeksi Kontur Dasar:</b> Mem

# ==========================================
# MENU 5: ANALISIS RESONANSI
# ==========================================
elif menu == "Analisis Resonansi":
    st.markdown('<div class="main-title">Analisis Risiko Resonansi Struktur Bangunan</div>', unsafe_allow_html=True)
    df = st.session_state.df_data
    if df is None:
        st.warning("Silakan upload data CSV terlebih dahulu di menu Upload Data.")
    else:
        num_floors = st.number_input("Masukkan Jumlah Lantai Bangunan yang Akan Disimulasikan (N):", min_value=1, max_value=50, value=3)
        fb = 10.0 / num_floors
        st.markdown(f'<div class="metric-box"><b>Estimasi Frekuensi Alami Bangunan (fb):</b> <h2>{fb:.2f} Hz</h2></div>', unsafe_allow_html=True)
        
        def evaluate_resonance(f0, fb):
            diff = abs(f0 - fb)
            if diff <= 0.5: return "Risiko Tinggi (Bahaya)"
            elif 0.5 < diff <= 1.5: return "Risiko Sedang (Waspada)"
            return "Risiko Rendah (Aman)"
            
        df_res = df[['Titik', 'f0', 'A0']].copy()
        df_res['fb (Freq Bangunan)'] = round(fb, 2)
        df_res['Selisih |f0 - fb| (Hz)'] = round(abs(df_res['f0'] - fb), 2)
        df_res['Status Risiko Resonansi'] = df_res['f0'].apply(lambda x: evaluate_resonance(x, fb))
        
        st.markdown('<div class="section-title">Tabel Hasil Perhitungan Skenario Resonansi Seismik</div>', unsafe_allow_html=True)
        st.dataframe(df_res, use_container_width=True)
