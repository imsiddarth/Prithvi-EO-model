import sys
sys.path.insert(0, "/home/sid/projects/BisagModels/prithvi_app")

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import torch, numpy as np, rasterio, os, time

from model     import load_model
from pipeline  import merge_bands, run_segmentation, convert_to_cog, get_bbox, detect_input_format
from geoserver import publish_cog, GEOSERVER
from database  import init_db, save_layer, get_all_layers

app = FastAPI(title="Prithvi-EO-300M API")

MODEL_PATH  = "/home/sid/projects/BisagModels/prithvi_300m_model/prithvi_300m_seg_best.pth"
NUM_CLASSES = 4
CLASS_NAMES = ["Vegetation", "Urban", "Water", "Barren"]
COG_STORE = "/mnt/c/prithvi_datasets/cog_store"  # ← Windows accessible
WORK_DIR    = "/home/sid/projects/BisagModels/prithvi_app/temp_work"

os.makedirs(COG_STORE, exist_ok=True)
os.makedirs(WORK_DIR,  exist_ok=True)
init_db()

print("Loading Prithvi-300M model...")
model = load_model(MODEL_PATH, NUM_CLASSES, is_segmentation=True)
print("✅ Model loaded!")

class PredictRequest(BaseModel):
    input_path  : str
    layer_name  : str
    class_names : list = CLASS_NAMES
    publish_wms : bool = True

@app.get("/")
def root():
    return {
        "status": "running",
        "model": "prithvi_eo_v2_300_tl",
        "classes": CLASS_NAMES,
        "geoserver_url": GEOSERVER
    }

@app.get("/health")
def health():
    return {"status": "healthy", "gpu": torch.cuda.is_available()}

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

processing_status = {}

def run_prediction_pipeline(req: PredictRequest, safe_layer_name: str):
    start = time.time()
    try:
        wsl_input_path = to_wsl_path(req.input_path)
        if not os.path.exists(wsl_input_path):
            processing_status[safe_layer_name] = {
                "status": "error",
                "message": f"Path not found: {req.input_path} (WSL: {wsl_input_path})",
                "steps": processing_status[safe_layer_name]["steps"] + ["❌ Input path not found on server."]
            }
            return

        processing_status[safe_layer_name]["steps"] = [
            f"📁 Loaded file/folder path: {req.input_path}",
            "🔄 Merging bands and generating unified TIFF..."
        ]

        merged_tif = os.path.join(WORK_DIR, f"{safe_layer_name}_merged.tif")
        seg_tif    = os.path.join(WORK_DIR, f"{safe_layer_name}_seg.tif")
        cog_path   = os.path.join(COG_STORE, f"{safe_layer_name}_cog.tif")

        merge_bands(wsl_input_path, merged_tif, WORK_DIR)

        processing_status[safe_layer_name]["steps"][-1] = "✅ Merged bands successfully"
        processing_status[safe_layer_name]["steps"].append("🛰️ Loading NASA-IBM Prithvi-300M model...")
        processing_status[safe_layer_name]["steps"].append("🛰️ Running patch-level segmentation (streaming in 64x64 chunks)...")

        def progress_cb(current, total):
            pct = 100 * current // total
            processing_status[safe_layer_name]["steps"][-1] = f"🛰️ Running model segmentation: {current}/{total} patches ({pct}%)"
            processing_status[safe_layer_name]["progress"] = pct

        run_segmentation(merged_tif, model, seg_tif, progress_cb=progress_cb)

        processing_status[safe_layer_name]["steps"][-1] = "✅ Model segmentation completed (all chunks compiled)"
        processing_status[safe_layer_name]["steps"].append("📦 Converting segmentation mask to Cloud Optimized GeoTIFF (COG)...")

        convert_to_cog(seg_tif, cog_path)

        processing_status[safe_layer_name]["steps"][-1] = "✅ Converted to Cloud Optimized GeoTIFF"

        with rasterio.open(cog_path) as src:
            mask = src.read(1)
        total = mask.size
        stats = {
            cls: {
                "pixels"  : int((mask == i).sum()),
                "percent" : round(float(100*(mask==i).sum()/total), 2),
                "area_km2": round(float((mask==i).sum()*0.01), 2)
            }
            for i, cls in enumerate(req.class_names)
        }

        wms_url = None
        if req.publish_wms:
            processing_status[safe_layer_name]["steps"].append("📡 Publishing WMS map layer to GeoServer...")
            wms_url = publish_cog(safe_layer_name, os.path.abspath(cog_path))
            processing_status[safe_layer_name]["steps"][-1] = "✅ Published layer to GeoServer"

        bbox, epsg = get_bbox(cog_path)
        save_layer(safe_layer_name, cog_path, wms_url or "", bbox, epsg)

        for f in [merged_tif, seg_tif]:
            if os.path.exists(f): os.remove(f)

        processing_status[safe_layer_name]["steps"].append("🎉 Completed!")
        processing_status[safe_layer_name]["status"] = "success"
        processing_status[safe_layer_name]["result"] = {
            "layer_name"     : safe_layer_name,
            "cog_path"       : cog_path,
            "wms_url"        : wms_url,
            "class_stats"    : stats,
            "processing_time": round(time.time()-start, 2)
        }
    except Exception as e:
        processing_status[safe_layer_name]["status"] = "error"
        processing_status[safe_layer_name]["message"] = str(e)
        processing_status[safe_layer_name]["steps"].append(f"❌ Error: {str(e)}")

@app.post("/predict")
def predict(req: PredictRequest, background_tasks: BackgroundTasks):
    wsl_path = to_wsl_path(req.input_path)
    if not os.path.exists(wsl_path):
        return {"status": "error", "message": f"Input path not found on server: {req.input_path} (WSL: {wsl_path})"}

    safe_layer_name = "".join([c if c.isalnum() or c == '_' else "_" for c in os.path.basename(req.layer_name)])
    safe_layer_name = "_".join([s for s in safe_layer_name.split("_") if s])
    if not safe_layer_name:
        safe_layer_name = f"layer_{int(time.time())}"

    processing_status[safe_layer_name] = {
        "status": "processing",
        "progress": 0,
        "steps": ["📡 Initializing prediction pipeline..."],
        "layer_name": safe_layer_name
    }

    background_tasks.add_task(run_prediction_pipeline, req, safe_layer_name)

    return {"status": "started", "layer_name": safe_layer_name}

@app.get("/status")
def get_status(layer_name: str):
    safe_layer_name = "".join([c if c.isalnum() or c == '_' else "_" for c in os.path.basename(layer_name)])
    safe_layer_name = "_".join([s for s in safe_layer_name.split("_") if s])
    if safe_layer_name not in processing_status:
        return {"status": "unknown"}
    return processing_status[safe_layer_name]

@app.get("/layers")
def get_layers():
    layers = get_all_layers()
    return {
        "layers": [
            {
                "id": l[0],
                "name": l[1],
                "wms_url": l[2],
                "bbox": l[3],
                "epsg": l[4],
                "created_at": l[5],
                "cog_path": l[6]
            }
            for l in layers
        ]
    }

@app.get("/browse")
def browse(path: str = "/"):
    if not path or path.strip() == "":
        path = "/"
    path = to_wsl_path(path)
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return {"status": "error", "message": f"Path '{path}' does not exist."}
    
    try:
        if not os.path.isdir(path):
            return {"status": "error", "message": f"Path '{path}' is not a directory."}
            
        parent_dir = os.path.dirname(path)
        if parent_dir == path:
            parent_dir = None
            
        items = []
        for name in os.listdir(path):
            full_p = os.path.join(path, name)
            is_dir = os.path.isdir(full_p)
            size = None
            if not is_dir:
                try:
                    size = os.path.getsize(full_p)
                except:
                    pass
            items.append({
                "name": name,
                "path": full_p,
                "is_dir": is_dir,
                "size": size
            })
            
        # Sort items: directories first, then files
        items.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
        
        return {
            "status": "success",
            "current_path": path,
            "parent_path": parent_dir,
            "items": items
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/detect")
def detect(path: str):
    if not path:
        return {"status": "error", "message": "No path provided"}
    path = to_wsl_path(path)
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return {"status": "error", "message": f"Path '{path}' does not exist."}
    try:
        fmt, detected_path = detect_input_format(path)
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