import os, glob, subprocess
import numpy as np
import rasterio
from rasterio.enums import Resampling
import torch

CLASS_NAMES  = ['Vegetation', 'Urban', 'Water', 'Barren']
PATCH_SIZE   = 64

def merge_bands(safe_folder, output_path):
    granule = glob.glob(f"{safe_folder}/GRANULE/*/IMG_DATA")[0]
    b02 = glob.glob(f"{granule}/R10m/*_B02*.jp2")[0]
    b03 = glob.glob(f"{granule}/R10m/*_B03*.jp2")[0]
    b04 = glob.glob(f"{granule}/R10m/*_B04*.jp2")[0]
    b08 = glob.glob(f"{granule}/R10m/*_B08*.jp2")[0]
    b11 = glob.glob(f"{granule}/R20m/*_B11*.jp2")[0]
    b12 = glob.glob(f"{granule}/R20m/*_B12*.jp2")[0]

    with rasterio.open(b02) as ref:
        meta   = ref.meta.copy()
        height = ref.height
        width  = ref.width

    meta.update(driver='GTiff', count=6, dtype='float32')

    with rasterio.open(output_path, 'w', **meta) as dst:
        for i, bp in enumerate([b02,b03,b04,b08,b11,b12], start=1):
            with rasterio.open(bp) as src:
                data = src.read(
                    1,
                    out_shape=(height, width),
                    resampling=Resampling.bilinear
                ).astype('float32')
            dst.write(data, i)

    return output_path, height, width

def run_segmentation(merged_tif, model, output_path):
    with rasterio.open(merged_tif) as src:
        img     = src.read().astype(np.float32) / 10000.0
        profile = src.profile.copy()
        H, W    = src.height, src.width

    full_mask = np.zeros((H, W), dtype=np.uint8)
    model.eval()

    with torch.no_grad():
        for row in range(0, H - PATCH_SIZE, PATCH_SIZE):
            for col in range(0, W - PATCH_SIZE, PATCH_SIZE):
                patch  = img[:, row:row+PATCH_SIZE, col:col+PATCH_SIZE]
                tensor = torch.tensor(patch).unsqueeze(0).cuda()
                out    = model(tensor)
                if hasattr(out, 'output'): out = out.output
                pred   = out.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
                full_mask[row:row+PATCH_SIZE, col:col+PATCH_SIZE] = pred

    profile.update(count=1, dtype='uint8')
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(full_mask, 1)

    return output_path

def convert_to_cog(input_tif, output_cog):
    subprocess.run([
        "gdal_translate",
        "-of", "COG",
        "-co", "COMPRESS=LZW",
        "-co", "TILED=YES",
        "-co", "BLOCKXSIZE=512",
        "-co", "BLOCKYSIZE=512",
        "-co", "OVERVIEWS=AUTO",
        "-co", "RESAMPLING=NEAREST",
        input_tif,
        output_cog
    ], check=True)
    return output_cog

def get_bbox(tif_path):
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        epsg   = src.crs.to_epsg()
    return bounds, f"EPSG:{epsg}"