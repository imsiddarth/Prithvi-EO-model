import streamlit as st
import requests
import json
import os
import re
import pandas as pd
import folium
from streamlit_folium import st_folium
from pyproj import Transformer
from PIL import Image
import io
import base64
import rasterio
import numpy as np

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Prithvi-EO Land Segmentation",
    page_icon="🛰️",
    layout="wide"
)

# ── Custom CSS Styling ────────────────────────────────────
def inject_custom_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Outfit', sans-serif !important;
}

/* Header styling */
.header-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 30px;
    border-radius: 16px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 4px 25px rgba(0, 0, 0, 0.2);
}

/* Title badge */
.header-title-badge {
    font-weight: 800;
    font-size: 2.2rem;
    background: linear-gradient(90deg, #3b82f6, #10b981);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0px;
    line-height: 1.2;
}

/* File browser box */
.explorer-box {
    background-color: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 18px;
}

/* Card design */
.card {
    background-color: rgba(255, 255, 255, 0.01);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 15px;
    transition: all 0.3s ease;
}
.card:hover {
    border-color: rgba(59, 130, 246, 0.3);
    transform: translateY(-2px);
}

/* Status Badges */
.status-pulse {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 8px;
    position: relative;
}
.status-pulse-online {
    background-color: #10b981;
    box-shadow: 0 0 10px #10b981;
}
.status-pulse-offline {
    background-color: #ef4444;
    box-shadow: 0 0 10px #ef4444;
}

/* Format Detection Alerts */
.detect-badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 5px;
    border: 1px solid transparent;
}
.detect-supported {
    background-color: rgba(16, 185, 129, 0.15);
    color: #10b981;
    border-color: rgba(16, 185, 129, 0.3);
}
.detect-unsupported {
    background-color: rgba(239, 68, 68, 0.15);
    color: #ef4444;
    border-color: rgba(239, 68, 68, 0.3);
}
.detect-unknown {
    background-color: rgba(245, 158, 11, 0.15);
    color: #f59e0b;
    border-color: rgba(245, 158, 11, 0.3);
}

/* Footer links */
.footer-container {
    margin-top: 60px;
    padding-top: 30px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
}
.footer-links-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
    margin-top: 18px;
    margin-bottom: 30px;
}
.footer-link-card {
    background-color: rgba(255, 255, 255, 0.01);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    transition: all 0.3s ease;
    text-decoration: none !important;
    color: #9ca3af !important;
    display: block;
}
.footer-link-card:hover {
    background-color: rgba(59, 130, 246, 0.05);
    border-color: rgba(59, 130, 246, 0.25);
    transform: translateY(-3px);
    color: #60a5fa !important;
}
.footer-link-icon {
    font-size: 1.6rem;
    margin-bottom: 8px;
    display: block;
}
.footer-link-title {
    font-weight: 600;
    font-size: 0.95rem;
    display: block;
    margin-bottom: 4px;
    color: #f3f4f6;
}
.footer-link-desc {
    font-size: 0.75rem;
    color: #6b7280;
}

/* Subsections */
.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 12px;
    color: #f3f4f6;
    border-left: 4px solid #3b82f6;
    padding-left: 10px;
}
</style>
""", unsafe_allow_html=True)

inject_custom_css()

# ── Session State Initialization ──────────────────────────
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "🛰️ Run Segmentation"
if "explorer_path" not in st.session_state:
    st.session_state.explorer_path = "/home/sid/projects/BisagModels"
if "selected_input_path" not in st.session_state:
    st.session_state.selected_input_path = ""
if "layer_name_val" not in st.session_state:
    st.session_state.layer_name_val = ""
if "active_wms_url" not in st.session_state:
    st.session_state.active_wms_url = ""
if "active_layer_name" not in st.session_state:
    st.session_state.active_layer_name = ""
if "active_bbox" not in st.session_state:
    st.session_state.active_bbox = ""
if "active_epsg" not in st.session_state:
    st.session_state.active_epsg = ""
if "active_cog_path" not in st.session_state:
    st.session_state.active_cog_path = ""
if "active_stats" not in st.session_state:
    st.session_state.active_stats = {}
if "class_labels" not in st.session_state:
    st.session_state.class_labels = ["Vegetation", "Urban", "Water", "Barren"]

# ── Sidebar Router ────────────────────────────────────────
st.sidebar.markdown('<div class="section-header">🧭 Navigation</div>', unsafe_allow_html=True)
page = st.sidebar.radio(
    "Go to page:",
    ["🛰️ Run Segmentation", "📂 Saved Layers History"],
    key="nav_page"
)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="section-header">⚙️ Connection & Info</div>', unsafe_allow_html=True)
BACKEND_URL = st.sidebar.text_input(
    "Backend API URL",
    "http://localhost:8001",
    key="backend_url_input"
)

# Server Health Check
server_status = "offline"
gpu_available = False
geoserver_url = "http://172.17.32.1:8080/geoserver" # default fallback
model_name = "Prithvi-EO-300M"

try:
    health_r = requests.get(f"{BACKEND_URL}/health", timeout=2)
    if health_r.status_code == 200 and health_r.json().get("status") == "healthy":
        server_status = "online"
        gpu_available = health_r.json().get("gpu", False)
        
    root_r = requests.get(f"{BACKEND_URL}/", timeout=2)
    if root_r.status_code == 200:
        root_data = root_r.json()
        geoserver_url = root_data.get("geoserver_url", geoserver_url)
        model_name = root_data.get("model", model_name)
except:
    server_status = "offline"

# Show Server Health Status Card in Sidebar
if server_status == "online":
    gpu_badge = "🟢 GPU Active" if gpu_available else "🟡 CPU Mode"
    st.sidebar.markdown(f"""
<div style="background-color: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.2); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
    <span class="status-pulse status-pulse-online"></span>
    <strong style="color: #10b981;">API Server Online</strong><br>
    <span style="font-size: 0.8rem; color: #9ca3af;">Model: {model_name}<br>{gpu_badge}</span>
</div>
""", unsafe_allow_html=True)
else:
    st.sidebar.markdown(f"""
<div style="background-color: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.2); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
    <span class="status-pulse status-pulse-offline"></span>
    <strong style="color: #ef4444;">API Server Offline</strong><br>
    <span style="font-size: 0.8rem; color: #9ca3af;">Cannot reach the backend. Operating in Local Fallback mode.</span>
</div>
""", unsafe_allow_html=True)

# ── Helper Functions ──────────────────────────────────────
def to_wsl_path(path_str):
    if not path_str:
        return path_str
    path_str = path_str.strip().strip('"').strip("'").strip()
    path_str = path_str.replace("\\", "/")
    import re
    match = re.match(r"^([a-zA-Z]):(?:/(.*))?$", path_str)
    if match:
        drive = match.group(1).lower()
        rest = match.group(2)
        if rest:
            return f"/mnt/{drive}/{rest}"
        else:
            return f"/mnt/{drive}"
    return path_str

def is_geo_file(filename):
    lower = filename.lower()
    return (
        lower.endswith(".safe") or 
        lower.endswith(".zip") or 
        lower.endswith(".tar") or 
        lower.endswith(".tif") or 
        lower.endswith(".tiff")
    )

def resolve_path_for_client(server_path):
    if not server_path:
        return server_path
    if os.name == 'posix': # WSL/Linux client
        return server_path
    # Windows client mapping
    if server_path.startswith("/mnt/c/"):
        return "C:/" + server_path[7:]
    elif server_path.startswith("/mnt/d/"):
        return "D:/" + server_path[7:]
    elif server_path.startswith("/home/sid/"):
        return "//wsl$/Ubuntu/home/sid/" + server_path[10:]
    return server_path

def get_directory_contents(path):
    wsl_path = to_wsl_path(path)
    try:
        r = requests.get(f"{BACKEND_URL}/browse", params={"path": wsl_path}, timeout=2)
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "success":
                return data
    except:
        pass
        
    try:
        wsl_path = os.path.abspath(wsl_path)
        if not os.path.exists(wsl_path):
            return {"status": "error", "message": f"Path '{wsl_path}' does not exist"}
        if not os.path.isdir(wsl_path):
            return {"status": "error", "message": f"Path '{wsl_path}' is not a directory"}
            
        parent_dir = os.path.dirname(wsl_path)
        if parent_dir == wsl_path:
            parent_dir = None
            
        items = []
        for name in os.listdir(wsl_path):
            full_p = os.path.join(wsl_path, name)
            is_dir = os.path.isdir(full_p)
            size = None
            if not is_dir:
                try: size = os.path.getsize(full_p)
                except: pass
            items.append({
                "name": name,
                "path": full_p,
                "is_dir": is_dir,
                "size": size
            })
        items.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
        return {
            "status": "success",
            "current_path": wsl_path,
            "parent_path": parent_dir,
            "items": items
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def detect_input_format(path):
    wsl_path = to_wsl_path(path)
    try:
        r = requests.get(f"{BACKEND_URL}/detect", params={"path": wsl_path}, timeout=2)
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "success":
                return data
    except:
        pass
        
    try:
        import sys
        sys.path.insert(0, "/home/sid/projects/BisagModels/prithvi_app")
        from pipeline import detect_input_format as local_detect
        fmt, detected_path = local_detect(wsl_path)
        descriptions = {
            'sentinel2_zip': "Sentinel-2 L2A Zipped Product (.zip)",
            'sentinel2_safe': "Sentinel-2 L2A SAFEs Folder (.SAFE)",
            'landsat_tar': "Landsat-8/9 Surface Reflectance TAR Archive (.tar)",
            'sentinel2_bands_folder': "Folder containing separate Sentinel-2 JP2/TIF bands",
            'landsat_bands_folder': "Folder containing separate Landsat-8/9 bands",
            'hls_folder': "Folder containing Harmonized Landsat Sentinel (HLS) bands",
            'merged_tif': "Multi-band GeoTIFF (>= 6 bands)",
            'single_band_tif': "Single-band GeoTIFF (< 6 bands, unsupported)",
            'unknown': "Unknown or unsupported format"
        }
        return {
            "status": "success",
            "format": fmt,
            "description": descriptions.get(fmt, "Unknown format"),
            "path": detected_path,
            "supported": fmt not in ('single_band_tif', 'unknown')
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def convert_bbox_to_latlon(bbox_str, epsg_str):
    if not bbox_str or "None" in epsg_str or not epsg_str:
        return None
    try:
        nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", bbox_str)]
        if len(nums) != 4:
            return None
            
        left, bottom, right, top = nums
        epsg_code = int(epsg_str.replace("EPSG:", ""))
        
        transformer = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)
        lon1, lat1 = transformer.transform(left, bottom)
        lon2, lat2 = transformer.transform(right, top)
        
        return [[min(lat1, lat2), min(lon1, lon2)], [max(lat1, lat2), max(lon1, lon2)]]
    except Exception as e:
        print(f"Error converting projection: {e}")
        return None

def parse_wms_details(wsl_wms_url):
    try:
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(wsl_wms_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        query = parse_qs(parsed.query)
        layer = query.get("LAYERS", [""])[0] or query.get("layers", [""])[0]
        return base_url, layer
    except Exception as e:
        return wsl_wms_url.split("?")[0], None

def compute_cog_stats(cog_path, class_names):
    resolved_path = resolve_path_for_client(cog_path)
    if not resolved_path or not os.path.exists(resolved_path):
        return {}
    try:
        with rasterio.open(resolved_path) as src:
            mask = src.read(1)
            x_res, y_res = src.res
            pixel_area_m2 = abs(x_res * y_res)
            
        total = mask.size
        stats = {}
        colors = ["#2ecc71", "#e74c3c", "#3498db", "#f1c40f"]
        emojis = ["🟩", "🟥", "🟦", "🟨"]
        for i, cls in enumerate(class_names):
            pixels = int((mask == i).sum())
            percent = round(float(100 * pixels / total), 2)
            area_m2 = pixels * pixel_area_m2
            area_km2 = round(area_m2 / 1_000_000, 3)
            area_ha = round(area_m2 / 10_000, 2)
            stats[cls] = {
                "pixels": pixels,
                "percent": percent,
                "area_km2": area_km2,
                "area_ha": area_ha,
                "color": colors[i % len(colors)],
                "emoji": emojis[i % len(emojis)]
            }
        return stats
    except Exception as e:
        print(f"Error computing stats from local raster: {e}")
        return {}

def render_local_geotiff_overlay(m, cog_path, fit_bounds_coords=None):
    resolved_path = resolve_path_for_client(cog_path)
    if not resolved_path or not os.path.exists(resolved_path):
        st.warning(f"Classification GeoTIFF file not found for local map overlay: {resolved_path}")
        return
        
    try:
        with rasterio.open(resolved_path) as src:
            mask = src.read(1)
            bounds = src.bounds
            epsg = src.crs.to_epsg()
            
        # Reproject bounds to lat/lon for folium overlay
        coords = convert_bbox_to_latlon(str(bounds), f"EPSG:{epsg}")
        if not coords:
            st.error("Failed to reproject local GeoTIFF coordinates for map overlay.")
            return
            
        # Resize mask for performance in browser overlay
        h, w = mask.shape
        max_dim = 1024
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img_pil = Image.fromarray(mask)
            img_pil = img_pil.resize((new_w, new_h), resample=Image.NEAREST)
            mask_resized = np.array(img_pil)
        else:
            mask_resized = mask
            
        # Create RGBA Image from indices
        rgba_data = np.zeros((mask_resized.shape[0], mask_resized.shape[1], 4), dtype=np.uint8)
        
        # Color mapping (Vegetation, Urban, Water, Barren)
        colors = [
            [46, 204, 113, 190],   # Vegetation (Green)
            [231, 76, 60, 190],    # Urban (Red)
            [52, 152, 219, 190],   # Water (Blue)
            [241, 196, 15, 190]    # Barren (Yellow)
        ]
        
        for val in range(4):
            rgba_data[mask_resized == val] = colors[val]
            
        img_out = Image.fromarray(rgba_data, 'RGBA')
        buffered = io.BytesIO()
        img_out.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        data_url = f"data:image/png;base64,{img_str}"
        
        # Add Folium image overlay
        folium.raster_layers.ImageOverlay(
            image=data_url,
            bounds=coords,
            opacity=0.75,
            name="Segmentation Result (Local Overlay)",
            interactive=True
        ).add_to(m)
        
        st.caption("🟢 Rendering classification overlay directly from GeoTIFF file...")
        
    except Exception as e:
        st.error(f"Error processing local GeoTIFF overlay: {e}")

# ── Dynamic Dashboard Rendering Component ──────────────────
def render_visualization_dashboard(class_names):
    if not st.session_state.active_layer_name:
        return
        
    st.markdown('<div class="section-header">📊 Active Visualization Dashboard</div>', unsafe_allow_html=True)
    st.info(f"Viewing active layer: **{st.session_state.active_layer_name}**")
    
    tab1, tab2, tab3 = st.tabs(["🗺️ Interactive Map Overlay", "📊 Land Cover Analysis", "📄 WMS & GIS Integration"])
    
    with tab1:
        st.subheader("Geographical Map Overlay")
        
        map_center = [20.5937, 78.9629] # Default India
        fit_bounds_coords = None
        
        if st.session_state.active_bbox and st.session_state.active_epsg:
            coords = convert_bbox_to_latlon(st.session_state.active_bbox, st.session_state.active_epsg)
            if coords:
                fit_bounds_coords = coords
                map_center = [
                    (coords[0][0] + coords[1][0]) / 2,
                    (coords[0][1] + coords[1][1]) / 2
                ]
        
        m = folium.Map(location=map_center, zoom_start=12, tiles="OpenStreetMap")
        
        # Decide if rendering via GeoServer WMS or Local TIFF File
        # Decide if rendering via GeoServer WMS or Local TIFF File
        use_local_fallback = True
        if st.session_state.active_wms_url and st.session_state.active_wms_url.strip() != "":
            base_wms, layer_fullname = parse_wms_details(st.session_state.active_wms_url)
            if layer_fullname:
                layer_name_only = layer_fullname.split(":")[-1] if ":" in layer_fullname else layer_fullname
                if "/" not in layer_name_only and "." not in layer_name_only and "\\" not in layer_name_only:
                    use_local_fallback = False
        
        if not use_local_fallback:
            base_wms, layer_fullname = parse_wms_details(st.session_state.active_wms_url)
            wms_layer = folium.WmsTileLayer(
                url=base_wms,
                layers=layer_fullname,
                fmt="image/png",
                transparent=True,
                name=st.session_state.active_layer_name,
                overlay=True,
                control=True,
                opacity=0.75
            )
            wms_layer.add_to(m)
            st.caption("🟢 Rendering WMS tile overlay from GeoServer...")
        else:
            render_local_geotiff_overlay(m, st.session_state.active_cog_path, fit_bounds_coords)
            
        legend_html = f"""
<div style="
    position: fixed; 
    bottom: 60px; 
    left: 20px; 
    width: 140px; 
    background-color: rgba(15, 23, 42, 0.95); 
    border: 1px solid rgba(255,255,255,0.15);
    color: #f3f4f6;
    z-index: 1000; 
    font-size: 11px;
    font-family: 'Outfit', sans-serif;
    padding: 10px;
    border-radius: 8px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.5);
">
    <strong style="display:block; margin-bottom:6px; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:3px;">Classification</strong>
    <i style="background:#2ecc71; width: 12px; height: 12px; float: left; margin-right: 8px; border-radius: 2px; border:1px solid rgba(255,255,255,0.2)"></i> {class_names[0]}<br>
    <i style="background:#e74c3c; width: 12px; height: 12px; float: left; margin-right: 8px; border-radius: 2px; border:1px solid rgba(255,255,255,0.2)"></i> {class_names[1]}<br>
    <i style="background:#3498db; width: 12px; height: 12px; float: left; margin-right: 8px; border-radius: 2px; border:1px solid rgba(255,255,255,0.2)"></i> {class_names[2]}<br>
    <i style="background:#f1c40f; width: 12px; height: 12px; float: left; margin-right: 8px; border-radius: 2px; border:1px solid rgba(255,255,255,0.2)"></i> {class_names[3]}<br>
</div>
"""
        m.get_root().html.add_child(folium.Element(legend_html))
        
        if fit_bounds_coords:
            m.fit_bounds(fit_bounds_coords)
            
        st_folium(m, width="100%", height=600)
        
    with tab2:
        st.subheader("Land Classification Statistics")
        
        # Read the file resolution and count area precisely
        stats = {}
        if st.session_state.active_cog_path:
            stats = compute_cog_stats(st.session_state.active_cog_path, class_names)
            
        # Fallback to session state stats if local file reading failed
        if not stats:
            stats = st.session_state.active_stats
            
        if stats:
            cols = st.columns(len(stats))
            for idx, (cls, data) in enumerate(stats.items()):
                with cols[idx]:
                    color = data.get("color", "#718096")
                    emoji = data.get("emoji", "🟩")
                    st.markdown(f"""
<div style="background-color: rgba(255, 255, 255, 0.02); border-left: 4px solid {color}; border: 1px solid rgba(255, 255, 255, 0.06); padding: 15px; border-radius: 8px; text-align: left;">
    <span style="font-size: 0.85rem; font-weight:600; color: #9ca3af;">{emoji} {cls}</span>
    <h2 style="margin: 5px 0 0 0; color: {color}; font-weight: 800; font-size:1.7rem;">{data['percent']}%</h2>
    <span style="font-size: 0.75rem; color: #6b7280;">
        <b>Pixels:</b> {data['pixels']:,} px<br>
        <b>Area:</b> {data['area_km2']:.3f} km²<br>
        <b>Hectares:</b> {data.get('area_ha', data['area_km2'] * 100):,.1f} ha
    </span>
</div>
""", unsafe_allow_html=True)
                    
            st.markdown("---")
            st.subheader("Distribution Chart")
            
            df = pd.DataFrame([
                {"Class": cls, "Coverage (%)": data["percent"]}
                for cls, data in stats.items()
            ])
            st.bar_chart(df.set_index("Class"))
        else:
            st.info("📊 Run a new classification or select a saved layer from history to compute specific pixel stats.")
            
    with tab3:
        st.subheader("Spatial Reference Metadata")
        st.json({
            "active_layer"   : st.session_state.active_layer_name,
            "bounding_box"   : st.session_state.active_bbox,
            "projection_epsg": st.session_state.active_epsg,
            "local_cog_path" : st.session_state.active_cog_path,
            "geoserver_wms"  : st.session_state.active_wms_url
        })
        
        st.markdown("---")
        
        # Details card explaining WMS usage
        if st.session_state.active_wms_url and st.session_state.active_wms_url.strip() != "":
            base_wms, layer_fullname = parse_wms_details(st.session_state.active_wms_url)
            
            st.markdown(f"""
### 🌐 About Web Map Service (WMS)
A Web Map Service (WMS) is an OGC (Open Geospatial Consortium) standard protocol for serving georeferenced map images over the internet. Instead of downloading raw geospatial data (which can be gigabytes in size), a WMS client requests a specific bounding box, and the server (GeoServer) renders it on-the-fly as a lightweight image (PNG/JPEG) to display on your screen.

#### 🛠️ How to use this WMS Layer in QGIS:
1. Open **QGIS**.
2. Go to the menu: **Layer ➡️ Add Layer ➡️ Add WMS/WMTS Layer...**
3. Click **New** to create a new server connection.
4. Name it (e.g. `Prithvi GeoServer`) and paste the **WMS Base URL** below:
   ```
   {base_wms}
   ```
5. Click **OK**, then click **Connect**.
6. Select the layer `{layer_fullname}` and click **Add**.

#### 🛠️ How to use this WMS Layer in ArcGIS Pro:
1. Open **ArcGIS Pro**.
2. Go to the **Insert** tab and click **Connections ➡️ New WMS Server**.
3. Paste the **WMS Base URL** and click **OK**.
4. The server connection will appear in your **Catalog** pane. Drag and drop the layer `{layer_fullname}` into your map.
""", unsafe_allow_html=True)
            
            st.markdown(f"📥 **WMS GetCapabilities**: [Download Layer Definition]({geoserver_url}/prithvi/wms?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetCapabilities)")
        else:
            st.markdown("""
### 🌐 Map Published Locally Only
This layer has not been published to GeoServer, so no external WMS links are generated. The map overlay above is rendered dynamically using the local GeoTIFF. 

To generate a shareable WMS link, please ensure you check the **Publish Result to GeoServer (WMS Layer)** checkbox before running the segmentation.
""")

# ──────────────────────────────────────────────────────────
# ── PAGE 1: RUN SEGMENTATION ──────────────────────────────
# ──────────────────────────────────────────────────────────
if page == "🛰️ Run Segmentation":
    # ── Main Layout: Page Title ────────────────────────────────
    st.markdown(f"""
<div class="header-card">
    <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 2px; color: #10b981; font-weight: 600;">Geospatial Machine Learning Portal</div>
    <h1 class="header-title-badge">🛰️ Prithvi-EO Land Segmentation</h1>
    <div style="color: #94a3b8; font-size: 1rem; margin-top: 8px;">
        Powered by NASA-IBM Prithvi-EO-300M Foundation Model. Select raw satellite archives or bands and perform pixel-level classification.
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">📁 Step 1: Browse & Select Server Files</div>', unsafe_allow_html=True)

    # Simplified path selector text input
    selected_path = st.text_input(
        "Input Folder or File Path (on server)",
        value=st.session_state.selected_input_path,
        placeholder="e.g. /home/sid/projects/BisagModels/test_data/my_image.SAFE",
        help="Paste the absolute path to your Sentinel-2 folder, .zip, Landsat .tar, or merged .tif file."
    )
    if selected_path != st.session_state.selected_input_path:
        st.session_state.selected_input_path = selected_path
        st.rerun()

    # Hiding the files listing inside an expander
    with st.expander("🔍 Browse Server Filesystem", expanded=False):
        path_bar_cols = st.columns([0.15, 0.15, 0.15, 0.55])
        with path_bar_cols[0]:
            if st.button("🏠 Home", use_container_width=True, help="Default Workspace"):
                st.session_state.explorer_path = "/home/sid/projects/BisagModels"
                st.rerun()
        with path_bar_cols[1]:
            if st.button("🗄️ Windows Mount", use_container_width=True, help="/mnt/c/prithvi_datasets"):
                st.session_state.explorer_path = "/mnt/c/prithvi_datasets"
                st.rerun()
        with path_bar_cols[2]:
            if st.button("⚙️ Root /", use_container_width=True, help="System Root"):
                st.session_state.explorer_path = "/"
                st.rerun()

        # Explorer browser
        current_path_input = st.text_input("Current Explorer Path", st.session_state.explorer_path)
        if current_path_input != st.session_state.explorer_path:
            st.session_state.explorer_path = current_path_input
            st.rerun()

        contents = get_directory_contents(st.session_state.explorer_path)

        if contents.get("status") == "error":
            st.error(f"❌ Error listing directory: {contents.get('message')}")
        else:
            # Breadcrumbs
            path_str = contents.get("current_path", "/")
            parts = [p for p in path_str.split("/") if p]
            
            st.markdown("**Breadcrumbs:**")
            bread_cols = st.columns(min(len(parts) + 1, 8))
            with bread_cols[0]:
                if st.button("📁 Root", key="bread_root"):
                    st.session_state.explorer_path = "/"
                    st.rerun()
                    
            visible_parts = parts[-6:]
            is_truncated = len(parts) > 6
            col_idx = 1
            
            if is_truncated:
                with bread_cols[col_idx]:
                    st.markdown("**...**")
                col_idx += 1
                
            for idx, part in enumerate(visible_parts):
                part_idx = len(parts) - len(visible_parts) + idx
                target_path = "/" + "/".join(parts[:part_idx+1])
                if col_idx < len(bread_cols):
                    with bread_cols[col_idx]:
                        if st.button(part, key=f"bread_{part_idx}", help=target_path):
                            st.session_state.explorer_path = target_path
                            st.rerun()
                    col_idx += 1

            st.markdown('<div class="explorer-box">', unsafe_allow_html=True)
            
            # Quick select active directory
            if st.button(f"🎯 Select Current Folder ({os.path.basename(st.session_state.explorer_path) or 'Root'})", key="select_current_folder", type="secondary"):
                st.session_state.selected_input_path = st.session_state.explorer_path
                base = os.path.basename(st.session_state.explorer_path) or "root"
                clean_name = "".join([c if c.isalnum() else "_" for c in base])
                st.session_state.layer_name_val = "_".join([s for s in clean_name.split("_") if s])
                st.rerun()

            # Search filter
            search_q = st.text_input("🔍 Filter items in current directory", "", key="explorer_filter")
            
            items = contents.get("items", [])
            filtered_items = [i for i in items if search_q.lower() in i["name"].lower()]
            
            parent_path = contents.get("parent_path")
            if parent_path:
                cols = st.columns([0.08, 0.77, 0.15])
                with cols[0]:
                    st.write("⬆️")
                with cols[1]:
                    if st.button(".. (Up One Level)", key="back_parent", use_container_width=True):
                        st.session_state.explorer_path = parent_path
                        st.rerun()
                st.markdown("---")
                
            if not filtered_items:
                st.info("No items found matching filter.")
            else:
                h_cols = st.columns([0.08, 0.52, 0.20, 0.20])
                with h_cols[1]: st.markdown("**Name**")
                with h_cols[2]: st.markdown("**Type / Size**")
                with h_cols[3]: st.markdown("**Action**")
                st.markdown("<hr style='margin: 5px 0 10px 0;'>", unsafe_allow_html=True)
                
                for idx, item in enumerate(filtered_items[:40]):
                    cols = st.columns([0.08, 0.52, 0.20, 0.20])
                    is_dir = item["is_dir"]
                    
                    icon = "📁" if is_dir else ("🛰️" if is_geo_file(item["name"]) else "📄")
                    with cols[0]:
                        st.write(icon)
                        
                    with cols[1]:
                        display_name = item["name"]
                        if len(display_name) > 60:
                            display_name = display_name[:57] + "..."
                        if is_dir:
                            if st.button(display_name, key=f"open_dir_{idx}", help=item["path"], use_container_width=True):
                                st.session_state.explorer_path = item["path"]
                                st.rerun()
                        else:
                            st.text(display_name)
                            
                    with cols[2]:
                        if is_dir:
                            st.caption("Folder")
                        else:
                            size = item.get("size")
                            if size is not None:
                                if size > 1024*1024: st.caption(f"{size/(1024*1024):.1f} MB")
                                elif size > 1024: st.caption(f"{size/1024:.1f} KB")
                                else: st.caption(f"{size} Bytes")
                            else:
                                st.caption("-")
                                
                    with cols[3]:
                        if st.button("🎯 Select", key=f"select_file_{idx}", help=f"Select {item['path']}", use_container_width=True):
                            st.session_state.selected_input_path = item["path"]
                            base = os.path.basename(item["path"])
                            if '.' in base:
                                base = base.rsplit('.', 1)[0]
                            clean_name = "".join([c if c.isalnum() else "_" for c in base])
                            st.session_state.layer_name_val = "_".join([s for s in clean_name.split("_") if s])
                            st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Format Detection Status Card ───────────────────────────
    if st.session_state.selected_input_path:
        st.markdown('<div class="section-header">🔍 Step 2: Selected Data Verification</div>', unsafe_allow_html=True)
        
        det = detect_input_format(st.session_state.selected_input_path)
        
        if det.get("status") == "error":
            st.error(f"Error checking format: {det.get('message')}")
        else:
            fmt = det.get("format", "unknown")
            desc = det.get("description", "Unknown")
            supported = det.get("supported", False)
            
            if supported:
                badge_class = "format-supported"
                status_text = "✅ Ready for processing!"
            else:
                badge_class = "format-unsupported"
                status_text = "⚠️ Format unsupported or needs merging. Large TIFFs (>= 6 bands) or SAFE directories are recommended."
                
            st.markdown(f"""
<div class="card">
    <strong>Selected Path on Server:</strong> <code style="font-size: 0.95rem; color:#3b82f6;">{st.session_state.selected_input_path}</code><br><br>
    <strong>Detected Input Format:</strong> 
    <span class="detect-badge {badge_class}">{desc}</span><br><br>
    <div style="font-weight: 600; font-size: 1.05rem;">{status_text}</div>
</div>
""", unsafe_allow_html=True)

    # ── Section 2: Parameters & Output Options ────────────────
    st.markdown('<div class="section-header">⚙️ Step 3: Configure Output & Class Names</div>', unsafe_allow_html=True)

    layer_name = st.text_input(
        "Output Layer Name",
        value=st.session_state.layer_name_val,
        placeholder="e.g. sentinel_delhi_segmentation",
        help="A unique name for storing outputs in GeoServer and database."
    )
    if layer_name != st.session_state.layer_name_val:
        st.session_state.layer_name_val = layer_name

    # Class Label Inputs
    st.subheader("🏷️ Map Class Labels")
    st.caption("Customize the display names for the 4 bands output by the Prithvi model.")
    c_cols = st.columns(4)
    with c_cols[0]: c1 = st.text_input("🟢 Class 1 Label", "Vegetation")
    with c_cols[1]: c2 = st.text_input("🔴 Class 2 Label", "Urban")
    with c_cols[2]: c3 = st.text_input("🔵 Class 3 Label", "Water")
    with c_cols[3]: c4 = st.text_input("🟡 Class 4 Label", "Barren")
    class_names = [c1, c2, c3, c4]

    # Check readiness
    ready = bool(st.session_state.selected_input_path.strip()) and bool(st.session_state.layer_name_val.strip())

    if not ready:
        st.warning("⚠️ Please select an input path and specify an output layer name to run.")

    st.markdown("---")

    publish_wms = st.checkbox(
        "📡 Publish Result to GeoServer (WMS Layer)",
        value=True,
        help="Saves output to Windows GeoServer store and applies custom land classification style."
    )

    # ── Execute Segmentation ──────────────────────────────────
    run_button = st.button("🚀 Run Segmentation Pipeline", disabled=not (ready and server_status == "online"), type="primary", use_container_width=True)

    if run_button and ready:
        progress = st.progress(0)
        status_log = st.status("Initializing Segmentation Pipeline...", expanded=True)
        
        try:
            status_log.write("📡 Connecting to Prithvi-EO server and verifying paths...")
            progress.progress(5)
            
            # Sanitize layer name to alphanumeric and underscores
            clean_layer_name = "".join([c if c.isalnum() or c == '_' else "_" for c in os.path.basename(st.session_state.layer_name_val.strip())])
            clean_layer_name = "_".join([s for s in clean_layer_name.split("_") if s])
            if not clean_layer_name:
                import time
                clean_layer_name = f"layer_{int(time.time())}"
                
            response = requests.post(
                f"{BACKEND_URL}/predict",
                json={
                    "input_path" : st.session_state.selected_input_path.strip(),
                    "layer_name" : clean_layer_name,
                    "class_names": class_names,
                    "publish_wms": publish_wms
                },
                timeout=10
            )
            
            if response.status_code == 200:
                init_data = response.json()
                if init_data.get("status") == "started":
                    st.session_state.class_labels = class_names
                    import time
                    while True:
                        try:
                            status_r = requests.get(f"{BACKEND_URL}/status", params={"layer_name": clean_layer_name}, timeout=3)
                            if status_r.status_code == 200:
                                status_data = status_r.json()
                                pipeline_status = status_data.get("status")
                                progress_val = status_data.get("progress", 0)
                                steps = status_data.get("steps", [])
                                
                                progress.progress(int(progress_val))
                                
                                status_log.empty()
                                with status_log:
                                    for s in steps:
                                        st.write(s)
                                
                                if pipeline_status == "success":
                                    progress.progress(100)
                                    result = status_data.get("result", {})
                                    status_log.update(label="✅ Pipeline Completed Successfully!", state="complete")
                                    st.success("🎉 Land cover segmentation is complete!")
                                    
                                    st.session_state.active_wms_url = result.get("wms_url") or ""
                                    st.session_state.active_layer_name = result.get("layer_name")
                                    st.session_state.active_cog_path = result.get("cog_path")
                                    st.session_state.active_stats = result.get("class_stats", {})
                                    
                                    try:
                                        r = requests.get(f"{BACKEND_URL}/layers")
                                        if r.status_code == 200:
                                            db_layers = r.json().get("layers", [])
                                            for db_l in db_layers:
                                                if db_l["name"] == result.get("layer_name"):
                                                    st.session_state.active_bbox = db_l["bbox"]
                                                    st.session_state.active_epsg = db_l["epsg"]
                                    except:
                                        pass
                                    
                                    st.rerun()
                                    break
                                elif pipeline_status == "error":
                                    progress.progress(0)
                                    status_log.update(label="❌ Pipeline Processing Error", state="error")
                                    st.error(f"Backend Server Error: {status_data.get('message', 'Unknown Error')}")
                                    break
                            else:
                                progress.progress(0)
                                status_log.update(label="❌ HTTP Connection Error", state="error")
                                st.error(f"Failed to communicate with API. Status code: {status_r.status_code}")
                                break
                        except Exception as e:
                            progress.progress(0)
                            status_log.update(label="❌ Connection Exception", state="error")
                            st.error(f"Connection error while polling status: {str(e)}")
                            break
                        time.sleep(1.5)
                else:
                    progress.progress(0)
                    status_log.update(label="❌ Pipeline Start Error", state="error")
                    st.error(f"Backend rejected prediction: {init_data.get('message', 'Unknown Error')}")
            else:
                progress.progress(0)
                status_log.update(label="❌ HTTP Connection Error", state="error")
                st.error(f"Failed to start pipeline. Status code: {response.status_code}")
        except Exception as e:
            progress.progress(0)
            status_log.update(label="❌ System Exception", state="error")
            st.error(f"Error starting segmentation: {str(e)}")


    # Class Label Inputs to allow runtime renaming
    st.subheader("🏷️ Customize Class Labels")
    c_cols = st.columns(4)
    with c_cols[0]: c1 = st.text_input("🟢 Class 1 Label", st.session_state.class_labels[0], key="v_c1")
    with c_cols[1]: c2 = st.text_input("🔴 Class 2 Label", st.session_state.class_labels[1], key="v_c2")
    with c_cols[2]: c3 = st.text_input("🔵 Class 3 Label", st.session_state.class_labels[2], key="v_c3")
    with c_cols[3]: c4 = st.text_input("🟡 Class 4 Label", st.session_state.class_labels[3], key="v_c4")
    class_names = [c1, c2, c3, c4]
    st.session_state.class_labels = class_names

    if st.session_state.active_layer_name:
        render_visualization_dashboard(class_names)
    else:
        st.info("📂 No active layer loaded. Please run a segmentation or select a layer from 'Saved Layers History' to visualize.")


# ──────────────────────────────────────────────────────────
# ── PAGE 3: SAVED LAYERS HISTORY ──────────────────────────
# ──────────────────────────────────────────────────────────
elif page == "📂 Saved Layers History":
    st.markdown("""
<div class="header-card">
    <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 2px; color: #3b82f6; font-weight: 600;">Geospatial Archive & WMS Server</div>
    <h1 class="header-title-badge">📂 Saved Layers History Repository</h1>
    <div style="color: #94a3b8; font-size: 1rem; margin-top: 8px;">
        Browse previously processed segmentation files, compute real-time spatial calculations, and retrieve WMS links for GIS tool integrations.
    </div>
</div>
""", unsafe_allow_html=True)

    # Class labels matching (Vegetation, Urban, Water, Barren)
    c_cols = st.columns(4)
    with c_cols[0]: c1 = st.text_input("🟢 Class 1 Label", "Vegetation", key="p2_c1")
    with c_cols[1]: c2 = st.text_input("🔴 Class 2 Label", "Urban", key="p2_c2")
    with c_cols[2]: c3 = st.text_input("🔵 Class 3 Label", "Water", key="p2_c3")
    with c_cols[3]: c4 = st.text_input("🟡 Class 4 Label", "Barren", key="p2_c4")
    class_names = [c1, c2, c3, c4]

    st.markdown('<div class="section-header">📁 Database Historical Records</div>', unsafe_allow_html=True)

    # Refresh Button
    if st.button("🔄 Refresh Database Listings", key="refresh_db"):
        st.cache_data.clear()

    # Load Layers
    layers = []
    try:
        r = requests.get(f"{BACKEND_URL}/layers", timeout=3)
        if r.status_code == 200:
            layers = r.json().get("layers", [])
    except:
        pass

    if not layers:
        st.info("No layers found in the database. Run a segmentation first on the prediction page!")
    else:
        # Display list of cards
        h_cols = st.columns([0.3, 0.4, 0.15, 0.15])
        with h_cols[0]: st.markdown("**Layer Name**")
        with h_cols[1]: st.markdown("**COG Path / Created At**")
        with h_cols[2]: st.markdown("**GeoServer WMS**")
        with h_cols[3]: st.markdown("**Action**")
        st.markdown("<hr style='margin: 5px 0 10px 0;'>", unsafe_allow_html=True)

        for idx, l in enumerate(layers):
            cols = st.columns([0.3, 0.4, 0.15, 0.15])
            has_wms = bool(l.get("wms_url") and l.get("wms_url").strip() != "")
            
            with cols[0]:
                st.markdown(f"**{l['name']}**")
                
            with cols[1]:
                date_str = l['created_at'][:10] + " " + l['created_at'][11:16]
                st.caption(f"📅 {date_str}")
                st.caption(f"💾 {l['cog_path']}")
                
            with cols[2]:
                if has_wms:
                    st.success("🟢 Published")
                else:
                    st.info("⚪ Local Only")
                    
            with cols[3]:
                if st.button("🗺️ Load Map", key=f"hist_load_{idx}", use_container_width=True):
                    st.session_state.active_wms_url = l["wms_url"]
                    st.session_state.active_layer_name = l["name"]
                    st.session_state.active_bbox = l["bbox"]
                    st.session_state.active_epsg = l["epsg"]
                    st.session_state.active_cog_path = l["cog_path"]
                    st.session_state.active_stats = {} 
                    st.session_state.class_labels = class_names
                    st.rerun()

        # Render dashboard if active layer is selected on the same page
        if st.session_state.active_layer_name:
            st.markdown("---")
            render_visualization_dashboard(class_names)


# ── Footer & External Running Services ────────────────────
st.markdown("""
<div class="footer-container">
<div style="font-weight: 700; font-size: 1.1rem; color: #f3f4f6; margin-bottom: 5px;">🔌 Active Infrastructure & Services</div>
<div style="font-size: 0.8rem; color: #6b7280; max-width: 600px; margin: 0 auto;">
The system relies on local API services and spatial servers. Access these external dashboards to monitor resource utilization, manage maps, or read system documentation.
</div>
<div class="footer-links-grid">
<a class="footer-link-card" href="http://localhost:8001/docs" target="_blank">
<span class="footer-link-icon">📖</span>
<span class="footer-link-title">Prithvi API Docs</span>
<span class="footer-link-desc">Interactive Swagger API documentation and testing interface.</span>
</a>
<a class="footer-link-card" href="http://172.17.32.1:8080/geoserver/web/" target="_blank">
<span class="footer-link-icon">🗺️</span>
<span class="footer-link-title">GeoServer Admin Console</span>
<span class="footer-link-desc">Manage geographical workspaces, coverage stores, and styles.</span>
</a>
<a class="footer-link-card" href="http://localhost:8001" target="_blank">
<span class="footer-link-icon">🔌</span>
<span class="footer-link-title">Uvicorn API Server</span>
<span class="footer-link-desc">Core FastAPI endpoint serving model inference and db requests.</span>
</a>
<a class="footer-link-card" href="https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M-sen1floods11" target="_blank">
<span class="footer-link-icon">🛰️</span>
<span class="footer-link-title">NASA-IBM Prithvi Model</span>
<span class="footer-link-desc">Explore the official foundation model repository and papers.</span>
</a>
</div>
<div style="font-size: 0.75rem; color: #4b5563; margin-bottom: 20px;">
Prithvi-EO Geospatial Suite • Designed by Advanced Agentic Coding Pair
</div>
</div>
""", unsafe_allow_html=True)