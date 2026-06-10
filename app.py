import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from scipy.spatial import distance_matrix

# ==========================================
# KONFIGURASI HALAMAN UTAMA
# ==========================================
st.set_page_config(
    page_title="Mikrozonasi Seismik HVSR",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-title { font-size: 32px; font-weight: bold; color: #1E3A8A; margin-bottom: 20px; }
    .section-title { font-size: 22px; font-weight: bold; color: #0D9488; margin-top: 20px; margin-bottom: 15px; }
    .metric-box { background-color: #F3F4F6; padding: 15px; border-radius: 10px; border-left: 5px solid #1E3A8A; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# GEOPHYSICAL FUNCTIONS
# ==========================================
def calculate_kg(a0, f0):
    return (a0 ** 2) / f0

def classify_kg(kg):
    if kg < 10: return "Rendah"
    elif 10 <= kg < 20: return "Sedang"
    else: return "Tinggi"

def classify_soil(f0):
    if f0 > 10: return "Batuan keras"
    elif 4 <= f0 <= 10: return "Tanah padat"
    elif 1 <= f0 < 4: return "Sedimen sedang"
    else: return "Sedimen tebal"

def color_picker_kg(status):
    if status == "Rendah": return "green"
    elif status == "Sedang": return "orange"
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
# SIDEBAR NAVIGATION & LOGO LOKAL FIX
# ==========================================
with st.sidebar:
    # Memanggil file logo lokal di Drive D
    try:
        # Menghapus batasan output vertikal agar gambar tampil full & proporsional
        st.image("logo_uin.png", width=130, channels="RGB")
    except:
        st.warning("Logo UIN (logo_uin.png) tidak ditemukan di Drive D")
    
    st.title("GEOFISIKA UIN SUKA")
    st.subheader("Analisis Mikrotremor HVSR")
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
    Aplikasi ini dirancang untuk memudahkan dalam melakukan pengolahan lanjut data parameter *Horizontal-to-Vertical Spectral Ratio* (HVSR) hasil pengukuran mikrotremor. 
    Melalui parameter frekuensi dominan ($f_0$) and amplifikasi ($A_0$), aplikasi ini mampu mengestimasi tingkat kerentanan tanah terhadap guncangan gempa bumi serta potensi resonansi dengan bangunan di atasnya.
    """)

# ==========================================
# MENU 2: UPLOAD DATA
# ==========================================
elif menu == "Upload Data":
    st.markdown('<div class="main-title">Upload Data Mikrotremor</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            required_cols = ['Titik', 'Longitude', 'Latitude', 'f0', 'A0']
            if all(col in df.columns for col in required_cols):
                df['Kg'] = calculate_kg(df['A0'], df['f0'])
                df['Tingkat Kerentanan'] = df['Kg'].apply(classify_kg)
                df['Karakteristik Tanah'] = df['f0'].apply(classify_soil)
                st.session_state.df_data = df
                st.success("Data berhasil diunggah!")
                st.dataframe(df, use_container_width=True)
            else:
                st.error("Format kolom CSV salah!")
        except Exception as e:
            st.error(f"Error: {e}")
    elif st.session_state.df_data is not None:
        st.dataframe(st.session_state.df_data, use_container_width=True)

# ==========================================
# MENU 3: ANALISIS KERENTANAN
# ==========================================
elif menu == "Analisis Kerentanan":
    st.markdown('<div class="main-title">Analisis Kerentanan Seismik & Statistik</div>', unsafe_allow_html=True)
    df = st.session_state.df_data
    if df is None:
        st.warning("Silakan upload data CSV terlebih dahulu.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1: st.markdown(f'<div class="metric-box"><b>Rata-rata f0</b><h2>{df["f0"].mean():.2f} Hz</h2></div>', unsafe_allow_html=True)
        with col2: st.markdown(f'<div class="metric-box"><b>Rata-rata A0</b><h2>{df["A0"].mean():.2f}</h2></div>', unsafe_allow_html=True)
        with col3: st.markdown(f'<div class="metric-box"><b>Rata-rata Kg</b><h2>{df["Kg"].mean():.2f}</h2></div>', unsafe_allow_html=True)
        
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            fig_hist = px.histogram(df, x="Kg", color="Tingkat Kerentanan", color_discrete_map={"Rendah": "green", "Sedang": "orange", "Tinggi": "red"})
            st.plotly_chart(fig_hist, use_container_width=True)
        with g_col2:
            fig_scatter = px.scatter(df, x="f0", y="A0", color="Tingkat Kerentanan", size="Kg", hover_name="Titik", color_discrete_map={"Rendah": "green", "Sedang": "orange", "Tinggi": "red"})
            st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# MENU 4: MIKROZONASI
# ==========================================
elif menu == "Mikrozonasi":
    st.markdown('<div class="main-title">Peta Mikrozonasi Seismik</div>', unsafe_allow_html=True)
    df = st.session_state.df_data
    if df is None:
        st.warning("Silakan upload data CSV terlebih dahulu.")
    else:
        m_col1, m_col2 = st.columns([3, 2])
        with m_col1:
            center_lat, center_lon = df['Latitude'].mean(), df['Longitude'].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
            for _, row in df.iterrows():
                popup_content = f"Titik: {row['Titik']}<br>Kg: {row['Kg']:.2f}"
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']], radius=8,
                    popup=folium.Popup(popup_content, max_width=200),
                    color=color_picker_kg(row['Tingkat Kerentanan']), fill=True, fill_opacity=0.7
                ).add_to(m)
            st_folium(m, width=600, height=450)
        with m_col2:
            x, y, z = df['Longitude'].values, df['Latitude'].values, df['Kg'].values
            xi = np.linspace(x.min() - 0.005, x.max() + 0.005, 100)
            yi = np.linspace(y.min() - 0.005, y.max() + 0.005, 100)
            xi, yi = np.meshgrid(xi, yi)
            zi = idw_interpolation(x, y, z, xi, yi, power=2)
            fig_idw = go.Figure(data=go.Contour(z=zi, x=np.linspace(x.min() - 0.005, x.max() + 0.005, 100), y=np.linspace(y.min() - 0.005, y.max() + 0.005, 100), colorscale='Jet'))
            st.plotly_chart(fig_idw, use_container_width=True)

# ==========================================
# MENU 5: ANALISIS RESONANSI
# ==========================================
elif menu == "Analisis Resonansi":
    st.markdown('<div class="main-title">Analisis Resonansi Bangunan</div>', unsafe_allow_html=True)
    df = st.session_state.df_data
    if df is None:
        st.warning("Silakan upload data CSV terlebih dahulu.")
    else:
        num_floors = st.number_input("Masukkan Jumlah Lantai Bangunan (N):", min_value=1, max_value=50, value=3)
        fb = 10.0 / num_floors
        st.metric(label="Estimasi Frekuensi Alami Bangunan (fb)", value=f"{fb:.2f} Hz")
        
        def evaluate_resonance(f0, fb):
            diff = abs(f0 - fb)
            if diff <= 0.5: return "Risiko Resonansi Tinggi"
            elif 0.5 < diff <= 1.5: return "Risiko Resonansi Sedang"
            return "Risiko Resonansi Rendah"
            
        df_res = df[['Titik', 'f0', 'A0']].copy()
        df_res['fb (Freq Bangunan)'] = round(fb, 2)
        df_res['Status Risiko Resonansi'] = df_res['f0'].apply(lambda x: evaluate_resonance(x, fb))
        st.dataframe(df_res, use_container_width=True)