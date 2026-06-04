import streamlit as st
import requests
import json

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Prithvi-EO Land Segmentation",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Prithvi-EO-300M Land Segmentation")
st.caption("Powered by NASA-IBM Prithvi Foundation Model")

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

BACKEND_URL = st.sidebar.text_input(
    "Backend API URL",
    "http://localhost:8001"
)

# Check server health
with st.sidebar:
    if st.button("🔌 Check Server"):
        try:
            r = requests.get(f"{BACKEND_URL}/health", timeout=5)
            data = r.json()
            if data["status"] == "healthy":
                st.success(f"✅ Server online\nGPU: {data['gpu']}")
            else:
                st.error("❌ Server unhealthy")
        except Exception as e:
            st.error(f"❌ Cannot reach server:\n{e}")

st.sidebar.markdown("---")

# ── Previous Layers ───────────────────────────────────────
st.sidebar.header("📂 Previous Layers")
if st.sidebar.button("🔄 Load Layers"):
    try:
        r = requests.get(f"{BACKEND_URL}/layers", timeout=5)
        layers = r.json().get("layers", [])
        if layers:
            for l in layers:
                st.sidebar.markdown(f"**{l['name']}**  \n{l['created_at'][:10]}")
        else:
            st.sidebar.info("No layers yet")
    except:
        st.sidebar.error("Could not load layers")

# ── Main Input Form ───────────────────────────────────────
st.header("📁 Input Data")

col1, col2 = st.columns(2)

with col1:
    input_path = st.text_input(
        "Input Image Path (on server)",
        placeholder="/data/ai_1/your_image.SAFE or .tif",
        help="Full path to your Sentinel-2 .SAFE folder, .zip, Landsat .tar, or merged .tif"
    )

with col2:
    layer_name = st.text_input(
        "Output Layer Name",
        placeholder="my_segmentation_01",
        help="Name for saving the result"
    )

# Class names
st.subheader("🏷️ Class Names")
col1, col2, col3, col4 = st.columns(4)
with col1: c1 = st.text_input("Class 1", "Vegetation")
with col2: c2 = st.text_input("Class 2", "Urban")
with col3: c3 = st.text_input("Class 3", "Water")
with col4: c4 = st.text_input("Class 4", "Barren")
class_names = [c1, c2, c3, c4]

# Options
publish_wms = st.checkbox("📡 Publish to GeoServer (WMS)", value=True)

# Validate
ready = bool(input_path.strip()) and bool(layer_name.strip())
if not ready:
    st.warning("⚠️ Please fill in both Input Path and Layer Name")

# ── Run Button ────────────────────────────────────────────
st.markdown("---")
run_button = st.button("🚀 Run Segmentation", disabled=not ready, type="primary")

if run_button and ready:
    progress = st.progress(0)
    status   = st.status("Starting...", expanded=True)

    try:
        status.write("🚀 Sending request to Prithvi server...")
        progress.progress(10)

        response = requests.post(
            f"{BACKEND_URL}/predict",
            json={
                "input_path" : input_path.strip(),
                "layer_name" : layer_name.strip(),
                "class_names": class_names,
                "publish_wms": publish_wms
            },
            timeout=600
        )

        progress.progress(90)
        result = response.json()

        if result["status"] == "success":
            progress.progress(100)
            status.update(label="✅ Complete!", state="complete")
            st.success("🎉 Segmentation complete!")

            # ── Results Tabs ──────────────────────────────
            tab1, tab2, tab3 = st.tabs(["📊 Class Statistics", "🗺️ WMS Map", "📄 Details"])

            with tab1:
                st.subheader("Land Cover Statistics")
                stats = result.get("class_stats", {})
                cols = st.columns(len(stats))
                for i, (cls, data) in enumerate(stats.items()):
                    with cols[i]:
                        st.metric(f"🟩 {cls}", f"{data['percent']}%")
                        st.caption(f"{data['pixels']:,} pixels\n{data['area_km2']} km²")

                # Bar chart
                if stats:
                    import pandas as pd
                    df = pd.DataFrame([
                        {"Class": cls, "Percent": data["percent"]}
                        for cls, data in stats.items()
                    ])
                    st.bar_chart(df.set_index("Class"))

            with tab2:
                wms_url = result.get("wms_url")
                if wms_url:
                    st.subheader("WMS Layer URL")
                    st.code(wms_url)
                    st.markdown(f"[🔗 Open in Browser]({wms_url})")
                else:
                    st.info("WMS not published (GeoServer may not be running)")

            with tab3:
                st.subheader("Processing Details")
                st.json({
                    "layer_name"     : result["layer_name"],
                    "cog_path"       : result["cog_path"],
                    "processing_time": f"{result['processing_time']} seconds"
                })

        else:
            progress.progress(0)
            status.update(label="❌ Error", state="error")
            st.error(f"Server error: {result.get('message', 'Unknown error')}")

    except requests.exceptions.Timeout:
        status.update(label="⏱️ Timeout", state="error")
        st.error("Request timed out. Large images can take 10+ minutes. Try again.")
    except Exception as e:
        status.update(label="❌ Error", state="error")
        st.error(f"Error: {str(e)}")