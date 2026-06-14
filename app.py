import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from scipy.spatial import distance_matrix

# ==========================================
# 1. KONFIGURASI HALAMAN UTAMA & LAYOUT
# ==========================================
st.set_page_config(
    page_title="Mikrozonasi Seismik HVSR - UIN Sunan Kalijaga",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk standardisasi Dashboard Geofisika UIN
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
# 2. GEOFISIKA & INTERPOLASI SPASIAL (IDW)
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

# Inisialisasi Session State Data CSV
if 'df_data' not in st.session_state:
    st.session_state.df_data = None

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    try:
        st.image("logo_uin.png", use_container_width=True)
    except:
        st.markdown("<h2 style='text-align: center; color: #1E3A8A;'>🕌 UIN SUKA</h2>", unsafe_allow_html=True)
            
    st.markdown("<h3 style='text-align: center; margin-top: 10px; color: #111827; font-size: 16px;'>GEOFISIKA UIN SUKA</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    menu = st.radio(
        "Pilih Menu Dashboard:",
        ["Analisis Kerentanan", "Mikrozonasi Spasial", "Analisis Resonansi"]
    )
    st.markdown("---")
    st.caption("UAS Seismologi - © 2026")

# ==========================================
# 4. HEADER UTAMA & UPLOAD DATA DIRECT (HOME)
# ==========================================
st.markdown('<div class="main-title">Aplikasi Mikrozonasi Seismik HVSR</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Program Studi Geofisika - UIN Sunan Kalijaga Yogyakarta</div>', unsafe_allow_html=True)

if st.session_state.df_data is None:
    st.info("👋 Selamat Datang! Silakan unggah file CSV hasil pengukuran lapangan Anda di bawah ini untuk memulai pemetaan kontur Surfer.")

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
            
            # Kalkulasi parameter geofisika
            df['Kg'] = calculate_kg(df['A0'], df['f0'])
            df['Tingkat Kerentanan'] = df['Kg'].apply(classify_kg)
            df['Karakteristik Tanah'] = df['f0'].apply(classify_soil)
            
            st.session_state.df_data = df
            st.success("Data CSV Lapangan Berhasil Diproses!")
        else:
            st.error("Format kolom CSV salah! Pastikan file memiliki nama kolom: TITIK, LONGITUDE, LATITUDE, F0, A0")
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")

df = st.session_state.df_data

# ==========================================
# MENU 1: ANALISIS KERENTANAN
# ==========================================
if menu == "Analisis Kerentanan":
    if df is None:
        st.warning("Silakan unggah file CSV data lapangan Anda terlebih dahulu.")
    else:
        st.markdown('<div class="section-title">Ringkasan Statistik Parameter Struktur Lapisan</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1: 
            st.markdown(f'<div class="metric-box"><b>Rata-rata f0 (Frekuensi Dominan)</b><h2>{df["f0"].mean():.2f} Hz</h2></div>', unsafe_allow_html=True)
        with col2: 
            st.markdown(f'<div class="metric-box"><b>Rata-rata A0 (Amplifikasi)</b><h2>{df["A0"].mean():.2f}</h2></div>', unsafe_allow_html=True)
        with col3: 
            st.markdown(f'<div class="metric-box"><b>Rata-rata Kg (Indeks Kerentanan)</b><h2>{df["Kg"].mean():.2f}</h2></div>', unsafe_allow_html=True)
        
        st.write("")
        st.dataframe(df[['Titik', 'Longitude', 'Latitude', 'f0', 'A0', 'Kg', 'Tingkat Kerentanan', 'Karakteristik Tanah']], use_container_width=True)
        
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            st.write("**Histogram Distribusi Jumlah Titik berdasarkan Nilai Kg**")
            fig_hist = go.Figure(data=[go.Histogram(x=df["Kg"], marker_color='#0D9488')])
            fig_hist.update_layout(xaxis_title="Indeks Kerentanan Seismik (Kg)", yaxis_title="Jumlah Titik", height=320)
            st.plotly_chart(fig_hist, use_container_width=True)
        with g_col2:
            st.write("**Scatter Plot Hubungan f0 vs A0 terhadap Nilai Kg**")
            fig_scatter = go.Figure(data=go.Scatter(
                x=df["f0"], y=df["A0"], mode='markers', 
                marker=dict(size=df["Kg"]*1.5, color=df["Kg"], colorscale='Jet', showscale=True)
            ))
            fig_scatter.update_layout(xaxis_title="Frekuensi Dominan f0 (Hz)", yaxis_title="Faktor Amplifikasi A0", height=320)
            st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# MENU 2: MIKROZONASI SPASIAL (4 OUTPUT MAPS DENGAN REPLIKA KONTUR SURFER)
# ==========================================
elif menu == "Mikrozonasi Spasial":
    if df is None:
        st.warning("Silakan unggah file CSV data lapangan Anda terlebih dahulu.")
    else:
        # PETA 1: PETA SEBARAN STASIUN UNTUK KERENTANAN SEISMIK
        st.markdown('<div class="section-title">Peta 1: GIS Sebaran Stasiun Ukur (Overlay Satelit & Popup Info)</div>', unsafe_allow_html=True)
        st.write("💡 *Klik pada marker lingkaran untuk melihat nomor titik, koordinat geografis, serta nilai Indeks Kerentanan Seismik (Kg).*")
        
        center_lat, center_lon = df['Latitude'].mean(), df['Longitude'].mean()
        m = folium.Map(
            location=[center_lat, center_lon], zoom_start=15,
            tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google Satellite Imagery'
        )
        
        for _, row in df.iterrows():
            popup_html = f"""
            <div style='font-family: Arial, sans-serif; font-size: 13px;'>
                <h4 style='margin:0 0 5px 0; color:#1E3A8A;'>Stasiun {row['Titik']}</h4>
                <b>Latitude:</b> {row['Latitude']:.5f}<br>
                <b>Longitude:</b> {row['Longitude']:.5f}<br><br>
                <span style='color:#0D9488;'><b>f₀:</b> {row['f0']:.2f} Hz</span><br>
                <span style='color:#EA580C;'><b>A₀:</b> {row['A0']:.2f}</span><br>
                <span style='background-color:#FEF2F2; padding:2px 5px; border-radius:3px;'><b>K_g:</b> {row['Kg']:.2f} ({row['Tingkat Kerentanan']})</span>
            </div>
            """
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']], radius=9,
                popup=folium.Popup(popup_html, max_width=250),
                color=color_picker_kg(row['Tingkat Kerentanan']), fill=True, fill_opacity=0.95
            ).add_to(m)
        st_folium(m, width=1100, height=400)
        
        # PROSES PEMBENTUKAN GRIDDING KONTUR SURFER (INTERPOLASI IDW)
        x, y = df['Longitude'].values, df['Latitude'].values
        x_line = np.linspace(x.min() - 0.002, x.max() + 0.002, 100)
        y_line = np.linspace(y.min() - 0.002, y.max() + 0.002, 100)
        xi, yi = np.meshgrid(x_line, y_line)
        
        zi_a0 = idw_interpolation(x, y, df['A0'].values, xi, yi)
        zi_f0 = idw_interpolation(x, y, df['f0'].values, xi, yi)
        zi_kg = idw_interpolation(x, y, df['Kg'].values, xi, yi)
        
        # Fungsi Plotter Pengganti Tampilan Surfer (Filled Contours 2D + Satelit Mapbox Alpha Layer)
        def create_surfer_2d_contour(zi_data, colorscale_name, title_lbl):
            fig = go.Figure()
            
            # 1. Overlay kontur padat berwarna (Surfer Style Fill Colors)
            fig.add_trace(go.Contour(
                z=zi_data, x=x_line, y=y_line,
                colorscale=colorscale_name,
                line=dict(color="black", width=0.8), # Kontur garis hitam halus seperti di Surfer
                contours=dict(showlabels=True, labelfont=dict(size=10, color="black")),
                colorbar=dict(title=title_lbl, thickness=15, titleside="right")
            ))
            
            # 2. Plotting Titik Posisi Lapangan
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="markers+text",
                text=df['Titik'].astype(str),
                textposition="top center",
                marker=dict(color="black", size=6, symbol="x"),
                name="Titik Ukur",
                showlegend=False
            ))
            
            # Layouting Bounding Box Koordinat presisi menyerupai plot cetak Surfer
            fig.update_layout(
                mapbox=dict(
                    style="satellite", # Integrasi basemap satelit di latar bawah kontur
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=14, layers=[]
                ),
                xaxis=dict(title="Longitude (Degree)", showgrid=True, gridcolor="rgba(0,0,0,0.1)", mirror=True, ticks="outside"),
                yaxis=dict(title="Latitude (Degree)", showgrid=True, gridcolor="rgba(0,0,0,0.1)", mirror=True, ticks="outside"),
                margin=dict(l=50, r=10, b=50, t=20),
                height=380, width=500,
                plot_bgcolor="white"
            )
            return fig

        st.markdown('<div class="section-title">Peta 2, 3, & 4: Output Model 2D Filled Contour Map (Surfer Replica Style)</div>', unsafe_allow_html=True)
        st.write("📈 *Tampilan visualisasi kontur di bawah mengadopsi fitur `Fill Colors` dan `Major/Minor Contours Line` Surfer dengan penempatan stasiun ukur (x).*")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("<h4 style='text-align: center; color: #1E3A8A; font-size:15px;'>Peta 2: Kontur Amplifikasi Situs (A₀ Map)</h4>", unsafe_allow_html=True)
            st.plotly_chart(create_surfer_2d_contour(zi_a0, "Viridis", "Nilai A₀"), use_container_width=True)
            
            st.markdown("<h4 style='text-align: center; color: #1E3A8A; font-size:15px;'>Peta 4: Kontur Kerentanan Seismik (Kg Map)</h4>", unsafe_allow_html=True)
            st.plotly_chart(create_surfer_2d_contour(zi_kg, "Jet", "Nilai K_g"), use_container_width=True)
            
        with col_m2:
            st.markdown("<h4 style='text-align: center; color: #1E3A8A; font-size:15px;'>Peta 3: Kontur Frekuensi Dominan Tanah (f₀ Map)</h4>", unsafe_allow_html=True)
            st.plotly_chart(create_surfer_2d_contour(zi_f0, "Plasma", "f₀ (Hz)"), use_container_width=True)
            
            # Mengosongkan kolom kanan bawah agar grid penempatan simetris rapi
            st.markdown("<br><br><div style='text-align:center; color:#6B7280; font-size:13px; border:2px dashed #E5E7EB; padding:60px; border-radius:10px;'><b>Sistem Gridding Interpolasi Surfer Terintegrasi</b><br>Berhasil memetakan batas koordinat minimum-maximum sesuai lampiran data lapangan.</div>", unsafe_allow_html=True)

# ==========================================
# MENU 3: ANALISIS RESONANSI
# ==========================================
elif menu == "Analisis Resonansi":
    if df is None:
        st.warning("Silakan unggah file CSV data lapangan Anda terlebih dahulu.")
    else:
        st.markdown('<div class="section-title">Analisis Risiko Resonansi Struktur Mikro Terhadap Bangunan Sekitar</div>', unsafe_allow_html=True)
        
        num_floors = st.number_input("Masukkan Jumlah Lantai Bangunan Target yang Disimulasikan (N):", min_value=1, max_value=40, value=3)
        fb = 10.0 / num_floors
        
        st.markdown(f'<div class="metric-box"><b>Karakteristik Frekuensi Alami Bangunan Rencana (fb):</b> <h2>{fb:.2f} Hz (Untuk Bangunan {num_floors} Lantai)</h2></div>', unsafe_allow_html=True)
        
        def evaluate_resonance_risk(f0, fb):
            diff = abs(f0 - fb)
            if diff <= 0.5: 
                return "Risiko Tinggi (Bahaya Resonansi)"
            elif 0.5 < diff <= 1.5: 
                return "Risiko Sedang (Waspada)"
            return "Risiko Rendah (Aman)"
            
        df_res = df[['Titik', 'Longitude', 'Latitude', 'f0', 'A0']].copy()
        df_res['fb (Freq Bangunan)'] = round(fb, 2)
        df_res['Selisih |f0 - fb| (Hz)'] = round(abs(df_res['f0'] - fb), 2)
        df_res['Kondisi Kerentanan Bangunan'] = df_res.apply(lambda row: evaluate_resonance_risk(row['f0'], fb), axis=1)
        
        st.write("")
        st.markdown("**Tabel Hasil Komparasi Parameter Frekuensi Dominan Tanah vs Bangunan (Baris-per-Baris):**")
        
        def color_rows(row):
            status = row['Kondisi Kerentanan Bangunan']
            if status == "Risiko Tinggi (Bahaya Resonansi)":
                return ['background-color: #FEE2E2; color: #991B1B'] * len(row)
            elif status == "Risiko Sedang (Waspada)":
                return ['background-color: #FEF3C7; color: #92400E'] * len(row)
            return ['background-color: #DCFCE7; color: #166534'] * len(row)
            
        styled_df = df_res.style.apply(color_rows, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        st.markdown("""
        <div class="ref-box">
        <b>Catatan Interpretasi Risiko Resonansi Struktur:</b><br>
        • <b>Risiko Tinggi (Merah):</b> Selisih frekuensi alami tanah ($f_0$) dan bangunan ($f_b$) sangat dekat ($\leq 0.5$ Hz). Bangunan rentan mengalami kerusakan fatal akibat fenomena resonansi gelombang seismik.<br>
        • <b>Risiko Sedang (Kuning):</b> Selisih frekuensi berada di rentang $0.5$ hingga $1.5$ Hz. Direkomendasikan menambah pengaku/perkuatan lateral pada struktur kolom utama.<br>
        • <b>Risiko Rendah (Hijau):</b> Selisih frekuensi $> 1.5$ Hz. Struktur aman karena karakteristik getaran tanah lokal dan fondasi gedung tidak saling menguatkan.
        </div>
        """, unsafe_allow_html=True)
