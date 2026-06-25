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
    .legend-item { font-size: 11px; color: #4B5563; margin-bottom: 3px; display: flex; align-items: center; }
    .dot { height: 8px; width: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; }
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
        return "Klas I (Tanah Keras)"
    elif 4 <= f0 < 6.7: 
        return "Klas II (Tanah Sedang)"
    elif 2.5 <= f0 < 4: 
        return "Klas III (Tanah Sedang-Lunak)"
    elif f0 < 2.5: 
        return "Klas IV (Tanah Lunak)"
    else: 
        return "Tanah Keras (>20 Hz)"

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
# 3. SIDEBAR NAVIGATION & KETERANGAN PARAMETER
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #1E3A8A; margin-top: 15px;'>🌋 GEOFISIKA</h2>", unsafe_allow_html=True)
    st.markdown("---")
    menu = st.radio("Pilih Menu Dashboard:", ["Analisis Kerentanan", "Mikrozonasi Spasial", "Analisis Resonansi"])
    st.markdown("---")
    
    # Keterangan Parameter di sudut kiri bawah seperti image_24c2e0.jpg
    st.markdown("""
    <div class="sidebar-desc-box">
        <div class="sidebar-desc-title">🌐 Keterangan Parameter</div>
        <div class="sidebar-desc-item"><b>📈 f0 (Hz):</b> Frekuensi dominan tanah hasil analisis perbandingan HVSR.</div>
        <div class="sidebar-desc-item"><b>📉 A0:</b> Amplifikasi maksimum atau faktor perbesaran gelombang situs.</div>
        <div class="sidebar-desc-item"><b>🛡️ Kg:</b> Indeks kerentanan seismik lapisan tanah makro ($Kg = A_0^2 / f_0$).</div>
        <div class="sidebar-desc-item"><b>🧱 Klas Tanah:</b> Klasifikasi jenis batuan berdasarkan periode getar Kanai (1957).</div>
    </div>
    """, unsafe_allow_html=True)

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
# MENU 1: ANALISIS KERENTANAN (DESAIN TARGET)
# ==========================================
if menu == "Analisis Kerentanan":
    if df is None:
        st.warning("Silakan unggah data lapangan CSV terlebih dahulu.")
    else:
        st.markdown('<div class="section-title">Ringkasan Statistik Lapisan</div>', unsafe_allow_html=True)
        
        # Hitung Nilai Rata-rata, Min, dan Max secara dinamis
        f0_avg, f0_min, f0_max = df["f0"].mean(), df["f0"].min(), df["f0"].max()
        a0_avg, a0_min, a0_max = df["A0"].mean(), df["A0"].min(), df["A0"].max()
        kg_avg, kg_min, kg_max = df["Kg"].mean(), df["Kg"].min(), df["Kg"].max()
        
        # Implementasi Kartu Statistik Sesuai Gaya Desain Target
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
        
        # Buat dataframe salinan khusus tampilan agar nama kolom rapi
        df_view = df.copy()
        df_view.insert(0, 'No.', range(1, len(df_view) + 1))
        
        # Fungsi pembantu formatting warna sel pandas dataframe styler
        def style_hvsr_table(val):
            # Warna untuk Klas Tanah
            if "Klas I" in str(val): return 'background-color: #DCFCE7; color: #15803D; font-weight: bold;'
            elif "Klas II" in str(val): return 'background-color: #E0F2FE; color: #0369A1; font-weight: bold;'
            elif "Klas III" in str(val): return 'background-color: #FEF3C7; color: #B45309; font-weight: bold;'
            elif "Klas IV" in str(val): return 'background-color: #FEE2E2; color: #B91C1C; font-weight: bold;'
            
            # Warna untuk Amplifikasi Situs & Kerentanan
            elif val in ["Low", "Rendah"]: return 'background-color: #DCFCE7; color: #166534; text-align: center;'
            elif val in ["Moderate", "Menengah"]: return 'background-color: #FEF3C7; color: #92400E; text-align: center;'
            elif val in ["High", "Tinggi", "Very High"]: return 'background-color: #FEE2E2; color: #991B1B; text-align: center;'
            return ''

        # Terapkan styling tabel dinamis ke kolom tertentu
        styled_table = df_view[[
            'No.', 'Titik', 'Longitude', 'Latitude', 'f0', 
            'Klasifikasi Klas Tanah (Kanai)', 'A0', 'Klasifikasi Amplifikasi Situs', 
            'Kg', 'Tingkat Kerentanan'
        ]].style.applymap(style_hvsr_table, subset=[
            'Klasifikasi Klas Tanah (Kanai)', 'Klasifikasi Amplifikasi Situs', 'Tingkat Kerentanan'
        ])
        
        st.dataframe(styled_table, use_container_width=True)
        
        # Menambahkan Blok Legenda Klasifikasi di bawah tabel sesuai gambar target
        st.markdown("""
        <div class="legend-box">
            <div class="row" style="display: flex; gap: 20px;">
                <div style="flex: 1.2;">
                    <div class="legend-col-title">Klasifikasi Klas Tanah (Kanai)</div>
                    <div class="legend-item"><span class="dot" style="background-color: #15803D;"></span><b>Klas I (6.7 - 20 Hz):</b> Tanah Keras / Batuan Padat</div>
                    <div class="legend-item"><span class="dot" style="background-color: #0369A1;"></span><b>Klas II (4.0 - 6.7 Hz):</b> Tanah Sedang / Aluvial Dangkal</div>
                    <div class="legend-item"><span class="dot" style="background-color: #B45309;"></span><b>Klas III (2.5 - 4.0 Hz):</b> Tanah Sedang-Lunak</div>
                    <div class="legend-item"><span class="dot" style="background-color: #B91C1C;"></span><b>Klas IV (&lt; 2.5 Hz):</b> Tanah Lunak / Sedimen Tebal</div>
                </div>
                <div style="flex: 0.9;">
                    <div class="legend-col-title">Klasifikasi Amplifikasi Situs (A0)</div>
                    <div class="legend-item"><span class="dot" style="background-color: #166534;"></span><b>Low:</b> A0 &lt; 3</div>
                    <div class="legend-item"><span class="dot" style="background-color: #92400E;"></span><b>Moderate:</b> 3 ≤ A0 &lt; 6</div>
                    <div class="legend-item"><span class="dot" style="background-color: #991B1B;"></span><b>High / Very High:</b> A0 ≥ 6</div>
                </div>
                <div style="flex: 0.9;">
                    <div class="legend-col-title">Tingkat Kerentanan (Kg)</div>
                    <div class="legend-item"><span class="dot" style="background-color: #166534;"></span><b>Rendah:</b> Kg &lt; 3</div>
                    <div class="legend-item"><span class="dot" style="background-color: #92400E;"></span><b>Menengah:</b> 3 ≤ Kg ≤ 6</div>
                    <div class="legend-item"><span class="dot" style="background-color: #991B1B;"></span><b>Tinggi:</b> Kg &gt; 6</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

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
                    popup_html = f"""
                    <div style='font-family: Arial, sans-serif; font-size: 12px; width: 195px;'>
                        <h5 style='margin:0 0 5px 0; color:#1E3A8A; border-bottom:1px solid #CCC; padding-bottom:3px;'><b>Titik: {row['Titik']}</b></h5>
                        <b>f0:</b> {row['f0']:.2f} Hz<br>
                        <b>A0:</b> {row['A0']:.2f}<br>
                        <b>Kg:</b> {row['Kg']:.2f}
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
            st_folium(generate_mikrozonasi_map(overlay_img=None, is_peta_1=True), width=1100, height=520, key="peta_1_perf_final")
        with tab2:
            st.markdown("#### Peta 2: Visualisasi Kontur Padat Faktor Amplifikasi Situs ($A_0$)")
            st_folium(generate_mikrozonasi_map(img_a0, is_peta_1=False), width=1100, height=520, key="peta_2_perf_final")
        with tab3:
            st.markdown("#### Peta 3: Visualisasi Kontur Padat Frekuensi Dominan Tanah ($f_0$)")
            st_folium(generate_mikrozonasi_map(img_f0, is_peta_1=False), width=1100, height=520, key="peta_3_perf_final")
        with tab4:
            st.markdown("#### Peta 4: Visualisasi Kontur Padat Indeks Kerentanan Seismik ($K_g$)")
            st_folium(generate_mikrozonasi_map(img_kg, is_peta_1=False), width=1100, height=520, key="peta_4_perf_final")
            
elif menu == "Analisis Resonansi":
    if df is None:
        st.warning("Silakan unggah data lapangan CSV terlebih dahulu.")
    else:
        st.markdown('<div class="section-title">Analisis Risiko Resonansi Struktur & Mikrotremor Tanah</div>', unsafe_allow_html=True)
        
        num_floors = st.number_input("Simulasi Jumlah Lantai Bangunan Rencana (N):", min_value=1, max_value=30, value=3)
        
        T = 0.1 * num_floors
        fb = 1.0 / T
        f0_mean = df["f0"].mean()
        
        col_r1, col_r2 = st.columns(2)
        col_r1.markdown(f'<div class="metric-box"><b>Frekuensi Alami Tanah (f₀) Rata-rata</b><h2>{f0_mean:.2f} Hz</h2></div>', unsafe_allow_html=True)
        col_r2.markdown(f'<div class="metric-box"><b>Frekuensi Alami Bangunan (f_b) Estimasian</b><h2>{fb:.2f} Hz</h2></div>', unsafe_allow_html=True)
        
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
