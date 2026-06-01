import streamlit as st
import os, zipfile, shutil
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import folium
from streamlit_folium import st_folium
from datetime import datetime

from model    import load_model
from pipeline import merge_bands, run_segmentation, convert_to_cog, get_bbox
from geoserver import publish_cog
from database import init_db, save_layer, get_all_layers

# ── Config ────────────────────────────────────────────────
MODEL_PATH  = "/data/ai_1/prithvi_project/model_300m/results/prithvi_300m_seg_best.pth"
COG_STORE   = "cog_store"
WORK_DIR    = "temp_work"
CLASS_NAMES  = ['Vegetation', 'Urban', 'Water', 'Barren']
CLASS_COLORS = ['#00b400', '#ff0000', '#0064ff', '#8b4513']

os.makedirs(COG_STORE, exist_ok=True)
os.makedirs(WORK_DIR,  exist_ok=True)
init_db()

# ── Page Setup ────────────────────────────────────────────
st.set_page_config(
    page_title = "Prithvi-EO-300M Land Use Classifier",
    page_icon  = "🛰️",
    layout     = "wide"
)

st.title("🛰️ Prithvi-EO-300M Land Use Segmentation")
st.markdown("Upload a Sentinel-2 `.SAFE` zip file to run segmentation and publish to GeoServer.")

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    geoserver_url  = st.text_input("GeoServer URL",  "http://localhost:8080/geoserver")
    geoserver_user = st.text_input("Username", "admin")
    geoserver_pass = st.text_input("Password", "geoserver", type="password")
    model_path     = st.text_input("Model Path", MODEL_PATH)

    st.divider()
    st.header("📋 Previous Layers")
    layers = get_all_layers()
    if layers:
        for lid, lname, wms, created in layers:
            st.text(f"🗂️ {lname}")
            st.caption(created[:19])
    else:
        st.info("No layers processed yet")

# ── Main Upload Area ──────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📁 Upload Sentinel-2 Data")
    uploaded_file = st.file_uploader(
        "Upload .SAFE zip file",
        type=["zip"],
        help="Upload the downloaded Sentinel-2 .SAFE.zip file"
    )

    if uploaded_file:
        st.success(f"✅ File received: {uploaded_file.name}")
        layer_name = st.text_input(
            "Layer Name",
            value=uploaded_file.name.split('.')[0][:20]
        )
        run_button = st.button("🚀 Run Segmentation", type="primary")

with col2:
    st.subheader("ℹ️ Pipeline Steps")
    st.markdown("""
    1. 📦 Extract .SAFE folder
    2. 🔗 Merge 6 spectral bands
    3. 🧠 Run Prithvi-300M segmentation
    4. 🗺️ Generate output .tif
    5. ⚡ Convert to COG
    6. 🌐 Publish to GeoServer
    7. 💾 Save to COG database
    8. 📍 Display WMS + map
    """)

# ── Processing ────────────────────────────────────────────
if uploaded_file and run_button:

    progress = st.progress(0)
    status   = st.status("Starting pipeline...", expanded=True)

    try:
        # Step 1 — Extract zip
        status.write("📦 Extracting .SAFE folder...")
        zip_path  = os.path.join(WORK_DIR, uploaded_file.name)
        safe_dir  = os.path.join(WORK_DIR, "extracted")
        os.makedirs(safe_dir, exist_ok=True)

        with open(zip_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(safe_dir)

        safe_folder = [
            os.path.join(safe_dir, d)
            for d in os.listdir(safe_dir)
            if d.endswith('.SAFE')
        ][0]
        progress.progress(15)

        # Step 2 — Merge bands
        status.write("🔗 Merging 6 spectral bands...")
        merged_tif = os.path.join(WORK_DIR, f"{layer_name}_merged.tif")
        merge_bands(safe_folder, merged_tif)
        progress.progress(30)

        # Step 3 — Load model
        status.write("🧠 Loading Prithvi-300M model...")
        model = load_model(model_path)
        progress.progress(45)

        # Step 4 — Run segmentation
        status.write("🔍 Running segmentation (this takes a few minutes)...")
        seg_tif = os.path.join(WORK_DIR, f"{layer_name}_segmentation.tif")
        run_segmentation(merged_tif, model, seg_tif)
        progress.progress(65)

        # Step 5 — Convert to COG
        status.write("⚡ Converting to Cloud Optimized GeoTIFF...")
        cog_path = os.path.join(COG_STORE, f"{layer_name}_cog.tif")
        convert_to_cog(seg_tif, cog_path)
        progress.progress(80)

        # Step 6 — Publish to GeoServer
        status.write("🌐 Publishing to GeoServer...")
        wms_url = publish_cog(layer_name, os.path.abspath(cog_path))
        progress.progress(90)

        # Step 7 — Save to database
        status.write("💾 Saving to COG database...")
        bbox, epsg = get_bbox(cog_path)
        save_layer(layer_name, cog_path, wms_url, bbox, epsg)
        progress.progress(100)

        status.update(label="✅ Pipeline complete!", state="complete")

        # ── Results ───────────────────────────────────────
        st.success("🎉 Segmentation complete!")

        tab1, tab2, tab3 = st.tabs(["🗺️ Map Preview", "🔗 WMS URL", "📥 Download COG"])

        with tab1:
            st.subheader("Segmentation Map")

            # Read COG and display
            with rasterio.open(cog_path) as src:
                mask    = src.read(1)
                bounds  = src.bounds
                # Center coordinates (convert from UTM to lat/lon approx)
                center_lat = (bounds.top + bounds.bottom) / 2
                center_lon = (bounds.left + bounds.right) / 2

            # Create colored map image
            COLOR_MAP = {
                0: [0,   180, 0  ],
                1: [200, 50,  50 ],
                2: [0,   100, 255],
                3: [210, 180, 140],
            }
            rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
            for cls, color in COLOR_MAP.items():
                rgb[mask == cls] = color

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(rgb)
            ax.axis('off')
            legend = [
                mpatches.Patch(color=np.array(c)/255, label=n)
                for c, n in zip(COLOR_MAP.values(), CLASS_NAMES)
            ]
            ax.legend(handles=legend, loc='lower right', fontsize=10)
            ax.set_title(f'Segmentation: {layer_name}', fontweight='bold')
            st.pyplot(fig)

            # Class distribution
            st.subheader("Class Distribution")
            cols = st.columns(4)
            total = mask.size
            for i, (cls, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
                count = (mask == i).sum()
                pct   = 100 * count / total
                cols[i].metric(
                    label=cls,
                    value=f"{pct:.1f}%",
                    delta=f"{count:,} pixels"
                )

        with tab2:
            st.subheader("WMS Service URLs")
            st.code(f"WMS GetMap:\n{wms_url}&BBOX={bbox.left},{bbox.bottom},{bbox.right},{bbox.top}", language="text")
            st.code(f"WMS GetCapabilities:\nhttp://localhost:8080/geoserver/prithvi/wms?SERVICE=WMS&REQUEST=GetCapabilities", language="text")
            st.code(f"WFS:\nhttp://localhost:8080/geoserver/prithvi/wfs?SERVICE=WFS&REQUEST=GetCapabilities", language="text")
            st.code(f"WCS:\nhttp://localhost:8080/geoserver/prithvi/wcs?SERVICE=WCS&REQUEST=GetCapabilities", language="text")
            st.code(f"WMTS:\nhttp://localhost:8080/geoserver/gwc/service/wmts?SERVICE=WMTS&REQUEST=GetCapabilities", language="text")

            st.info("Add WMS URL to QGIS: Layer → Add WMS/WMTS Layer → paste URL above")

        with tab3:
            st.subheader("Download Cloud Optimized GeoTIFF")
            with open(cog_path, 'rb') as f:
                st.download_button(
                    label     = "📥 Download COG",
                    data      = f,
                    file_name = f"{layer_name}_cog.tif",
                    mime      = "image/tiff"
                )
            st.caption(f"File saved at: {cog_path}")

        # Cleanup temp files
        shutil.rmtree(safe_dir, ignore_errors=True)
        os.remove(zip_path)
        os.remove(merged_tif)
        os.remove(seg_tif)

    except Exception as e:
        status.update(label="❌ Error occurred", state="error")
        st.error(f"Error: {str(e)}")
        st.exception(e)