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
    if 6.7 <= f0 <= 20: return "Klas I (Tanah Keras / Batuan Padat)"
    elif 4 <= f0 < 6.7: return "Klas II (Tanah Sedang / Aluvial Dangkal)"
    elif 2.5 <= f0 < 4: return "Klas III (Tanah Sedang-Lunak)"
    elif f0 < 2.5: return "Klas IV (Tanah Lunak / Sedimen Tebal)"
    else: return "Di luar Rentang Standar Kanai (>20 Hz)"

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
        st.dataframe(df[['Titik', 'Longitude', 'Latitude', 'f0', 'Klasifikasi Klas Tanah (Kanai)', 'A0', 'Klasifikasi Amplifikasi Situs', 'Kg', 'Tingkat Kerentanan']], use_container_width=True)

elif menu == "Mikrozonasi Spasial":
    if df is None:
        st.warning("Silakan unggah data lapangan CSV terlebih dahulu.")
    else:
        st.markdown('<div class="section-title">Output Peta Mikrozonasi Spasial (Google Satellite)</div>', unsafe_allow_html=True)
        
        center_lat, center_lon = df['Latitude'].mean(), df['Longitude'].mean()
        pad = 0.002  
        min_lat, max_lat = df['Latitude'].min() - pad, df['Latitude'].max() + pad
        min_lon, max_lon = df['Longitude'].min() - pad, df['Longitude'].max() + pad
        bounds = [[min_lat, min_lon], [max_lat, max_lon]]
        
        x, y = df['Longitude'].values, df['Latitude'].values
        x_line = np.linspace(min_lon, max_lon, 250)
        y_line = np.linspace(min_lat, max_lat, 250)
        xi, yi = np.meshgrid(x_line, y_line)
        
        zi_a0 = idw_interpolation(x, y, df['A0'].values, xi, yi)
        zi_f0 = idw_interpolation(x, y, df['f0'].values, xi, yi)
        zi_kg = idw_interpolation(x, y, df['Kg'].values, xi, yi)
        
        img_a0 = create_surfer_contour_overlay(xi, yi, zi_a0, 'rainbow') 
        img_f0 = create_surfer_contour_overlay(xi, yi, zi_f0, 'viridis')
        img_kg = create_surfer_contour_overlay(xi, yi, zi_kg, 'jet')     
        
        def generate_mikrozonasi_map(overlay_img=None, is_peta_1=False):
            m = folium.Map(
                location=[center_lat, center_lon], zoom_start=16,
                tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                attr='Google Satellite Imagery',
                control_scale=True  
            )
            
            if overlay_img is not None:
                folium.raster_layers.ImageOverlay(
                    image=overlay_img,
                    bounds=bounds,
                    opacity=0.60,  
                    mercator_project=True
                ).add_to(m)
            
            if is_peta_1:
                for _, row in df.iterrows():
                    # Menampilkan teks nama kelas tanah secara utuh di pop-up peta
                    popup_html = f"""
                    <div style='font-family: Arial, sans-serif; font-size: 12px; width: 190px;'>
                        <h5 style='margin:0 0 5px 0; color:#1E3A8A; border-bottom:1px solid #CCC; padding-bottom:3px;'><b>Titik Ke: {row['Titik']}</b></h5>
                        <b>f0:</b> {row['f0']:.2f} Hz<br>
                        <b>A0:</b> {row['A0']:.2f}<br>
                        <b>Kg:</b> {row['Kg']:.2f}<br>
                        <b>Situs:</b> {row['Klasifikasi Klas Tanah (Kanai)']}
                    </div>
                    """
                    folium.CircleMarker(
                        location=[row['Latitude'], row['Longitude']], radius=5,
                        popup=folium.Popup(popup_html, max_width=250),
                        color='black', weight=1.2,
                        fill=True, fill_color=color_picker_kg(row['Tingkat Kerentanan']), fill_opacity=1.0
                    ).add_to(m)
                    
            return m

        tab1, tab2, tab3, tab4 = st.tabs([
            "📌 Peta 1: Kerentanan Seismik (Titik Koordinat)", 
            "🟢 Peta 2: Peta Kontur Amplifikasi ($A_0$)", 
            "🟣 Peta 3: Peta Kontur Frekuensi Dominan ($f_0$)", 
            "🔴 Peta 4: Peta Kontur Indeks Kerentanan ($K_g$)"
        ])
        
        with tab1:
            st.markdown("#### Peta 1: Sebaran Spasial Titik Pengukuran Lapangan")
            st_folium(generate_mikrozonasi_map(overlay_img=None, is_peta_1=True), width=1100, height=520, key="peta_1_final_rev")
        with tab2:
            st.markdown("#### Peta 2: Visualisasi Kontur Padat Faktor Amplifikasi Situs ($A_0$)")
            st_folium(generate_mikrozonasi_map(img_a0, is_peta_1=False), width=1100, height=520, key="peta_2_final_rev")
        with tab3:
            st.markdown("#### Peta 3: Visualisasi Kontur Padat Frekuensi Dominan Tanah ($f_0$)")
            st_folium(generate_mikrozonasi_map(img_f0, is_peta_1=False), width=1100, height=520, key="peta_3_final_rev")
        with tab4:
            st.markdown("#### Peta 4: Visualisasi Kontur Padat Indeks Kerentanan Seismik ($K_g$)")
            st_folium(generate_mikrozonasi_map(img_kg, is_peta_1=False), width=1100, height=520, key="peta_4_final_rev")
            
elif menu == "Analisis Resonansi":
    if df is None:
        st.warning("Silakan unggah data lapangan CSV terlebih dahulu.")
    else:
        st.markdown('<div class="section-title">Analisis Risiko Resonansi Struktur & Mikrotremor Tanah</div>', unsafe_allow_html=True)
        
        num_floors = st.number_input("Simulasi Jumlah Lantai Bangunan Rencana (N):", min_value=1, max_value=30, value=3)
        
        T = 0.1 * num_floors
        fb = 1.0 / T
        f0_mean = df["f0"].mean()
        
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.markdown(f'<div class="metric-box"><b>Frekuensi Alami Tanah (f₀) Rata-rata</b><h2>{f0_mean:.2f} Hz</h2></div>', unsafe_allow_html=True)
        col_r2.markdown(f'<div class="metric-box"><b>Frekuensi Alami Bangunan (f_b) Estimasian</b><h2>{fb:.2f} Hz</h2></div>', unsafe_allow_html=True)
        
        selisih_kumulatif = abs(f0_mean - fb) / fb
        if selisih_kumulatif < 0.10:
            status_txt, warna_lbl = "⚠️ TINGGI (Sangat Bahaya)", "red"
        elif selisih_kumulatif < 0.30:
            status_txt, warna_lbl = "🔸 MENENGAH (Waspada)", "orange"
        else:
            status_txt, warna_lbl = "✅ RENDAH (Kondisi Aman)", "green"
            
        col_r3.markdown(f'<div class="metric-box"><b>Potensi Bahaya Resonansi Wilayah</b><h3 style="color:{warna_lbl};">{status_txt}</h3></div>', unsafe_allow_html=True)
        
        st.write("")
        fig, ax = plt.subplots(figsize=(6, 2.5))
        y_pos = ['Frekuensi Bangunan ($f_b$)', 'Frekuensi Getar Tanah ($f_0$)']
        values = [fb, f0_mean]
        bar_colors = ['#1E3A8A', '#0D9488']
        
        bars = ax.barh(y_pos, values, color=bar_colors, height=0.45)
        ax.set_xlabel('Frekuensi (Hz)', fontsize=9)
        ax.set_xlim(0, max(max(values) + 3, 10))
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax.tick_params(labelsize=9)
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.15, bar.get_y() + bar.get_height()/2, f'{width:.2f} Hz', 
                    va='center', ha='left', fontsize=9, fontweight='bold')
                    
        st.pyplot(fig)
        plt.close(fig)
        
        st.write("")
        st.markdown("**Tabel Deteksi Tingkat Kerentanan Resonansi Bangunan per Stasiun Ukur:**")
        
        df_res = df.copy()
        df_res["f_b (Hz)"] = round(fb, 2)
        df_res["Selisih Rasio"] = round(abs(df_res["f0"] - fb) / fb, 3)
        df_res["Risiko Resonansi"] = df_res["Selisih Rasio"].apply(lambda r: "Tinggi" if r < 0.10 else ("Sedang" if r < 0.30 else "Rendah"))
        
        def highlight_resonance_rows(row):
            status = row['Risiko Resonansi']
            if status == "Tinggi":
                return ['background-color: #FEE2E2; color: #991B1B'] * len(row)
            elif status == "Sedang":
                return ['background-color: #FEF3C7; color: #92400E'] * len(row)
            return ['background-color: #DCFCE7; color: #166534'] * len(row)
            
        styled_df = df_res[['Titik', 'Longitude', 'Latitude', 'f0', 'f_b (Hz)', 'Risiko Resonansi']].style.apply(highlight_resonance_rows, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        st.markdown("""
        <div class="ref-box">
        <b>Pedoman Interpretasi Bahaya Rekayasa Resonansi Seismik:</b><br>
        • <b>Risiko Tinggi (Merah):</b> Selisih frekuensi getar alami tanah ($f_0$) dan struktur ($f_b$) sangat rapat (Rasio selisih < 10%). Gedung rentan mengalami kehancuran struktural hebat akibat getaran gelombang gempa yang teramplifikasi ekstrem.<br>
        • <b>Risiko Sedang (Kuning):</b> Rentang rasio berada di angka 10% - 30%. Disarankan melakukan penguatan konstruksi kolom dan fondasi lateral bangunan.<br>
        • <b>Risiko Rendah (Hijau):</b> Rasio selisih > 30%. Struktur bangunan aman karena karakteristik getar tanah lokal tidak memicu penguatan simpangan gedung.
        </div>
        """, unsafe_allow_html=True)
