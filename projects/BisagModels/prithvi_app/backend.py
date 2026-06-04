import sys
sys.path.insert(0, "/home/sid/projects/BisagModels/prithvi_app")

from fastapi import FastAPI
from pydantic import BaseModel
import torch, numpy as np, rasterio, os, time

from model     import load_model
from pipeline  import merge_bands, run_segmentation, convert_to_cog, get_bbox
from geoserver import publish_cog
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
    return {"status": "running", "model": "prithvi_eo_v2_300_tl", "classes": CLASS_NAMES}

@app.get("/health")
def health():
    return {"status": "healthy", "gpu": torch.cuda.is_available()}

@app.post("/predict")
def predict(req: PredictRequest):
    start = time.time()
    if not os.path.exists(req.input_path):
        return {"status": "error", "message": f"Path not found: {req.input_path}"}
    try:
        merged_tif = os.path.join(WORK_DIR, f"{req.layer_name}_merged.tif")
        seg_tif    = os.path.join(WORK_DIR, f"{req.layer_name}_seg.tif")
        cog_path   = os.path.join(COG_STORE, f"{req.layer_name}_cog.tif")

        merge_bands(req.input_path, merged_tif, WORK_DIR)
        run_segmentation(merged_tif, model, seg_tif)
        convert_to_cog(seg_tif, cog_path)

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
            wms_url = publish_cog(req.layer_name, os.path.abspath(cog_path))

        bbox, epsg = get_bbox(cog_path)
        save_layer(req.layer_name, cog_path, wms_url or "", bbox, epsg)

        for f in [merged_tif, seg_tif]:
            if os.path.exists(f): os.remove(f)

        return {
            "status"         : "success",
            "layer_name"     : req.layer_name,
            "cog_path"       : cog_path,
            "wms_url"        : wms_url,
            "class_stats"    : stats,
            "processing_time": round(time.time()-start, 2)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/layers")
def get_layers():
    layers = get_all_layers()
    return {"layers": [{"id":l[0],"name":l[1],"wms_url":l[2],"created_at":l[3]} for l in layers]}