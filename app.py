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
    page_title="Mikrozonasi Seismik HVSR",
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
# SIDEBAR NAVIGATION & LOGO AMAN
# ==========================================
with st.sidebar:
    try:
        # Membaca file logo_uin.png dari repository GitHub Anda
        st.image("logo_uin.png", use_container_width=True)
    except:
        # Cadangan teks rapi jika file logo belum terdeteksi sistem git online
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
                st.error(f"Format salah. Pastikan kolom berisi: TITIK, LONGITUDE, LATITUDE, F0, A0")
        except Exception as e:
            st.error(f"Error: {e}")
    elif st.session_state.df_data is not None:
        st.dataframe(st.session_state.df_data, use_container_width=True)

# ==========================================
# MENU 3: ANALISIS KERENTANAN (FIXED ERROR BUG)
# ==========================================
elif menu == "Analisis Kerentanan":
    st.markdown('<div class="main-title">Analisis Kerentanan Seismik & Statistik Detail</div>', unsafe_allow_html=True)
    df = st.session_state.df_data
    if df is None:
        st.warning("Silakan upload data CSV terlebih dahulu di menu Upload Data.")
    else:
        # Menampilkan box metrik rata-rata secara rapi
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
        
        # Grafik analisis distribusi data spasial
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

        st.markdown("""
        <div class="ref-box">
        <b>Referensi Standar Ambang Batas Klasifikasi Kerentanan Seismik (Nakamura, 1997):</b><br>
        • <b>Rendah (Kg < 3):</b> Tingkat deformasi tanah rendah, batuan penyusun cenderung kompak/keras.<br>
        • <b>Menengah (3 ≤ Kg ≤ 6):</b> Tingkat deformasi tanah sedang, batuan penyusun berupa aluvium/sedimen sedang.<br>
        • <b>Tinggi (Kg > 6):</b> Tingkat deformasi tanah tinggi, sangat rawan mengalami pelunakan/amplifikasi parah akibat gempa bumi.
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# MENU 4: MIKROZONASI (DENGAN BASEMAP SURFACE & TEXTURE)
# ==========================================
elif menu == "Mikrozonasi":
    st.markdown('<div class="main-title">Peta Kerentanan Seismik & Model Spasial 3D</div>', unsafe_allow_html=True)
    df = st.session_state.df_data
    if df is None:
        st.warning("Silakan upload data CSV terlebih dahulu di menu Upload Data.")
    else:
        # PETA INTERAKTIF UTAMA DENGAN BASEMAP TERRAIN (PERMUKAAN BUMI RIIL)
        st.markdown('<div class="section-title">Peta 1: Overlay Parameter Seismik di Atas Basemap Permukaan Bumi (Terrain)</div>', unsafe_allow_html=True)
        st.write("Peta di bawah ini menggunakan basemap khusus untuk memperlihatkan kontur relief permukaan bumi asli (lembah dan bukit) di area penelitian.")
        
        center_lat, center_lon = df['Latitude'].mean(), df['Longitude'].mean()
        
        # Menggunakan basemap Terrain agar tekstur permukaan bumi terlihat jelas
        m = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=14,
            tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
            attr='&copy; OpenTopoMap contributors'
        )
        
        # Menambahkan pilihan basemap Satelit Google jika ingin melihat permukaan foto udara
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
            attr='Google Satellite',
            name='Google Satellite',
            overlay=False
        ).add_to(m)
        
        for _, row in df.iterrows():
            tooltip_info = f"Stasiun: {row['Titik']}"
            popup_html = f"""
            <div style='font-family: Arial, sans-serif; font-size: 12px; width: 220px; line-height: 1.4;'>
                <h4 style='margin:0 0 5px 0; color:#1E3A8A;'>Stasiun {row['Titik']}</h4>
                <hr style='border:none; border-top:1px solid #CCC; margin:5px 0;'>
                <b>Koordinat:</b> {row['Latitude']:.5f}, {row['Longitude']:.5f}<br>
                <b>Frekuensi (f₀):</b> {row['f0']:.2f} Hz<br>
                <b>Amplifikasi (A₀):</b> {row['A0']:.2f}<br>
                <b>Nilai Kerentanan (K_g):</b> {row['Kg']:.2f}<br>
                <b>Status Risiko:</b> <span style='color:{color_picker_kg(row['Tingkat Kerentanan'])}; font-weight:bold;'>{row['Tingkat Kerentanan']}</span><br>
                <b>Karakteristik:</b> {row['Karakteristik Tanah']}
            </div>
            """
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']], radius=10,
                popup=folium.Popup(popup_html, max_width=250), tooltip=tooltip_info,
                color='black', weight=1,
                fill_color=color_picker_kg(row['Tingkat Kerentanan']), fill_opacity=0.9
            ).add_to(m)
            
        folium.LayerControl().add_to(m)
        st_folium(m, width=1100, height=450)
        st.caption("💡 *Petunjuk: Anda bisa mengubah tampilan permukaan ke mode Satelit Riil atau Topografi menggunakan tombol layer di pojok kanan atas peta.*")
        
        # PROSES INTERPOLASI IDW GRID
        x, y = df['Longitude'].values, df['Latitude'].values
        x_line = np.linspace(x.min() - 0.003, x.max() + 0.003, 100)
        y_line = np.linspace(y.min() - 0.003, y.max() + 0.003, 100)
        xi, yi = np.meshgrid(x_line, y_line)
        
        zi_f0 = idw_interpolation(x, y, df['f0'].values, xi, yi)
        zi_a0 = idw_interpolation(x, y, df['A0'].values, xi, yi)
        zi_kg = idw_interpolation(x, y, df['Kg'].values, xi, yi)
        
        # VISUALISASI 3 PETA 3D SURFACE SEJAJAR DENGAN GRID TEKSTUR RAPAT (SIMULASI BASMEAP)
        st.markdown('<div class="section-title">Peta 2: 3D Surface Model dengan Efek Tekstur Kontur Permukaan</div>', unsafe_allow_html=True)
        
        col_3a, col_3b, col_3c = st.columns(3)
        
        # Optimasi shading & pencahayaan ekstrem agar guratan tekstur permukaan menonjol keluar
        high_relief_lighting = dict(
            ambient=0.5,
            diffuse=0.9,
            fresnel=0.4,
            specular=0.6,
            roughness=0.2
        )
        
        # Pengaturan garis kontur badan jaring agar membentuk bayangan relief bergaris rapat
        surface_contours = dict(
            x=dict(show=True, start=x_line.min(), end=x_line.max(), size=(x_line.max()-x_line.min())/20, color="

# ==========================================
# MENU 5: ANALISIS RESONANSI
# ==========================================
elif menu == "Analisis Resonansi":
    st.markdown('<div class="main-title">Analisis Risiko Resonansi Struktur Bangunan</div>', unsafe_allow_html=True)
    df = st.session_state.df_data
    if df is None:
        st.warning("Silakan upload data CSV terlebih dahulu di menu Upload Data.")
    else:
        st.markdown("""
        > ℹ️ **Catatan Keilmuan (Estimasi Empiris):** Penentuan nilai Frekuensi Alami Bangunan ($f_b$) di bawah ini merupakan bentuk **Estimasi Matematis** menggunakan rumus empiris standar *Building Seismic Safety Council* (BSSC). Sifatnya berfungsi sebagai indikasi awal kebencanaan (skrining awal) sebelum dilakukan pengujian langsung menggunakan instrumen akselerometer pada struktur bangunan gedung.
        """)
        
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
        
        st.markdown("""
        <div class="ref-box">
        <b>Referensi Kebencanaan Sipil-Geofisika (Standardisasi Matriks Risiko Resonansi):</b><br>
        • <b>Risiko Tinggi (Selisih ≤ 0.5 Hz):</b> Frekuensi getaran tanah penutup berhimpit dengan frekuensi struktur gedung. Gedung terancam runtuh akibat amplifikasi ayunan resonansi ekstrem.<br>
        • <b>Risiko Sedang (0.5 Hz < Selisih ≤ 1.5 Hz):</b> Efek getaran kopel terjadi namun tidak mencapai amplitudo destruktif maksimal.<br>
        • <b>Risiko Rendah (Selisih > 1.5 Hz):</b> Aman. Struktur gedung aman dari fenomena resonansi karena perbedaan respons frekuensi getar yang kontras.
        </div>
        """, unsafe_allow_html=True)
