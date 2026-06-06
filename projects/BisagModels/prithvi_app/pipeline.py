import os, glob, subprocess, tarfile, zipfile
import numpy as np
import rasterio
from rasterio.enums import Resampling
import torch

CLASS_NAMES = ['Vegetation', 'Urban', 'Water', 'Barren']
PATCH_SIZE  = 64

# ── Format Detection ──────────────────────────────────────

def detect_input_format(input_path):
    """
    Detects what format the input data is in
    Returns format name and relevant info
    """
    # Case 1 — Zipped .SAFE file
    if input_path.endswith('.SAFE.zip') or (
        input_path.endswith('.zip') and 'MSIL2A' in input_path
    ):
        return 'sentinel2_zip', input_path

    # Case 2 — .SAFE folder
    if input_path.endswith('.SAFE') and os.path.isdir(input_path):
        return 'sentinel2_safe', input_path

    # Case 3 — Landsat .tar bundle
    if input_path.endswith('.tar') and os.path.isfile(input_path):
        return 'landsat_tar', input_path

    # Case 4 — Folder with separate files
    if os.path.isdir(input_path):
        files = os.listdir(input_path)

        # Sentinel-2 separate bands
        has_b02 = any('B02' in f or 'B2' in f for f in files)
        has_b08 = any('B08' in f or 'B8' in f for f in files)
        if has_b02 and has_b08:
            return 'sentinel2_bands_folder', input_path

        # Landsat separate bands
        has_sr_b2 = any('SR_B2' in f for f in files)
        has_sr_b5 = any('SR_B5' in f for f in files)
        if has_sr_b2 and has_sr_b5:
            return 'landsat_bands_folder', input_path

        # HLS format
        has_hls = any('HLS' in f for f in files)
        if has_hls:
            return 'hls_folder', input_path

    # Case 5 — Already merged multi-band .tif
    if input_path.endswith('.tif') or input_path.endswith('.tiff'):
        with rasterio.open(input_path) as src:
            band_count = src.count
        if band_count >= 6:
            return 'merged_tif', input_path
        else:
            return 'single_band_tif', input_path

    return 'unknown', input_path


# ── Band Merging — All Formats ────────────────────────────

def merge_bands(input_path, output_path, work_dir="temp_work"):
    """
    Detects format and merges 6 bands into single .tif
    Works for all input formats
    """
    os.makedirs(work_dir, exist_ok=True)
    fmt, path = detect_input_format(input_path)
    print(f"Detected format: {fmt}")

    if fmt == 'sentinel2_zip':
        return _merge_sentinel2_zip(path, output_path, work_dir)

    elif fmt == 'sentinel2_safe':
        return _merge_sentinel2_safe(path, output_path)

    elif fmt == 'landsat_tar':
        return _merge_landsat_tar(path, output_path, work_dir)

    elif fmt == 'sentinel2_bands_folder':
        return _merge_sentinel2_bands_folder(path, output_path)

    elif fmt == 'landsat_bands_folder':
        return _merge_landsat_bands_folder(path, output_path)

    elif fmt == 'hls_folder':
        return _merge_hls_folder(path, output_path)

    elif fmt == 'merged_tif':
        # Already merged — just copy
        import shutil
        shutil.copy(path, output_path)
        return output_path, None, None

    else:
        raise ValueError(f"Unknown or unsupported input format: {input_path}")


# ── Individual Format Handlers ────────────────────────────

def _merge_sentinel2_safe(safe_folder, output_path):
    """Handles .SAFE folder"""
    granule = glob.glob(f"{safe_folder}/GRANULE/*/IMG_DATA")[0]

    b02 = glob.glob(f"{granule}/R10m/*_B02*.jp2")[0]
    b03 = glob.glob(f"{granule}/R10m/*_B03*.jp2")[0]
    b04 = glob.glob(f"{granule}/R10m/*_B04*.jp2")[0]
    b08 = glob.glob(f"{granule}/R10m/*_B08*.jp2")[0]
    b11 = glob.glob(f"{granule}/R20m/*_B11*.jp2")[0]
    b12 = glob.glob(f"{granule}/R20m/*_B12*.jp2")[0]

    return _stack_bands([b02,b03,b04,b08,b11,b12], output_path)


def _merge_sentinel2_zip(zip_path, output_path, work_dir):
    """Handles .SAFE.zip file"""
    extract_dir = os.path.join(work_dir, "sentinel2_extracted")
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)

    safe_folder = glob.glob(f"{extract_dir}/*.SAFE")[0]
    return _merge_sentinel2_safe(safe_folder, output_path)


def _merge_sentinel2_bands_folder(folder, output_path):
    """Handles folder with separate Sentinel-2 band files"""
    files = os.listdir(folder)

    def find_band(pattern):
        matches = [
            os.path.join(folder, f) for f in files
            if pattern in f and (f.endswith('.jp2') or f.endswith('.tif'))
        ]
        if not matches:
            raise FileNotFoundError(f"Band {pattern} not found in {folder}")
        return matches[0]

    b02 = find_band('B02')
    b03 = find_band('B03')
    b04 = find_band('B04')
    b08 = find_band('B08')
    b11 = find_band('B11')
    b12 = find_band('B12')

    return _stack_bands([b02,b03,b04,b08,b11,b12], output_path)


def _merge_landsat_tar(tar_path, output_path, work_dir):
    """Handles Landsat .tar bundle"""
    extract_dir = os.path.join(work_dir, "landsat_extracted")
    os.makedirs(extract_dir, exist_ok=True)

    with tarfile.open(tar_path) as tar:
        tar.extractall(extract_dir)

    return _merge_landsat_bands_folder(extract_dir, output_path)


def _merge_landsat_bands_folder(folder, output_path):
    """Handles folder with separate Landsat band files"""
    files = os.listdir(folder)

    def find_band(pattern):
        matches = [
            os.path.join(folder, f) for f in files
            if pattern in f and f.endswith('.TIF')
        ]
        if not matches:
            raise FileNotFoundError(f"Band {pattern} not found in {folder}")
        return matches[0]

    b2 = find_band('SR_B2')   # BLUE
    b3 = find_band('SR_B3')   # GREEN
    b4 = find_band('SR_B4')   # RED
    b5 = find_band('SR_B5')   # NIR
    b6 = find_band('SR_B6')   # SWIR1
    b7 = find_band('SR_B7')   # SWIR2

    return _stack_bands([b2,b3,b4,b5,b6,b7], output_path, norm_factor=55000.0)


def _merge_hls_folder(folder, output_path):
    """Handles HLS format folder"""
    files = os.listdir(folder)

    def find_band(band_name):
        matches = [
            os.path.join(folder, f) for f in files
            if f.endswith(f"{band_name}.tif") or f.endswith(f"{band_name}.TIF")
        ]
        if not matches:
            raise FileNotFoundError(f"HLS band {band_name} not found")
        return matches[0]

    b02 = find_band('B02')
    b03 = find_band('B03')
    b04 = find_band('B04')
    b8a = find_band('B8A')   # NIR Narrow in HLS
    b11 = find_band('B11')
    b12 = find_band('B12')

    return _stack_bands([b02,b03,b04,b8a,b11,b12], output_path)


# ── Core Band Stacking Function ───────────────────────────
def _stack_bands(band_paths, output_path, norm_factor=10000.0):
    """
    Takes list of band file paths and stacks into single .tif
    Resamples all bands to match the first band's resolution
    """
    with rasterio.open(band_paths[0]) as ref:
        meta   = ref.meta.copy()
        height = ref.height
        width  = ref.width

    meta.update(driver='GTiff', count=6, dtype='float32')

    with rasterio.open(output_path, 'w', **meta) as dst:
        for i, bp in enumerate(band_paths, start=1):
            with rasterio.open(bp) as src:
                data = src.read(
                    1,
                    out_shape   = (height, width),
                    resampling  = Resampling.bilinear
                ).astype('float32')
            dst.write(data, i)

    return output_path, height, width


# ── Segmentation ──────────────────────────────────────────

def run_segmentation(merged_tif, model, output_path, chunk_size=1024, progress_cb=None):
    with rasterio.open(merged_tif) as src:
        profile = src.profile.copy()
        H, W    = src.height, src.width

        # Count total patches first for progress tracking
        total_patches = 0
        for row in range(0, H - PATCH_SIZE, PATCH_SIZE):
            for col in range(0, W - PATCH_SIZE, PATCH_SIZE):
                total_patches += 1

        print(f"Image size: {H}x{W}")
        print(f"Total patches: {total_patches}")

        profile.update(count=1, dtype='uint8')
        
        # Open output file for writing chunk-by-chunk
        with rasterio.open(output_path, 'w', **profile) as dst:
            stripe_rows = 1024
            done = 0
            model.eval()

            with torch.no_grad():
                for s_row in range(0, H - PATCH_SIZE, stripe_rows):
                    # We process vertical patches starting from s_row up to s_row + stripe_rows
                    patch_rows = []
                    for r in range(s_row, min(s_row + stripe_rows, H - PATCH_SIZE), PATCH_SIZE):
                        patch_rows.append(r)
                    
                    if not patch_rows:
                        continue

                    # If this is the last stripe, we read all remaining rows to avoid gaps at the bottom
                    is_last_stripe = (s_row + stripe_rows >= H - PATCH_SIZE)
                    if is_last_stripe:
                        h_read = H - s_row
                    else:
                        h_read = (patch_rows[-1] + PATCH_SIZE) - s_row

                    # Read window from source (only 6 bands for this window)
                    window = rasterio.windows.Window(0, s_row, W, h_read)
                    img_stripe = src.read(window=window).astype(np.float32) / 10000.0

                    # Create empty mask for this stripe
                    stripe_mask = np.zeros((h_read, W), dtype=np.uint8)

                    # Predict patches within the stripe
                    for r in patch_rows:
                        r_rel = r - s_row
                        for col in range(0, W - PATCH_SIZE, PATCH_SIZE):
                            patch  = img_stripe[:, r_rel:r_rel+PATCH_SIZE, col:col+PATCH_SIZE]
                            tensor = torch.tensor(patch).unsqueeze(0).cuda()
                            out    = model(tensor)
                            if hasattr(out, 'output'): out = out.output
                            pred   = out.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
                            stripe_mask[r_rel:r_rel+PATCH_SIZE, col:col+PATCH_SIZE] = pred

                            # Free GPU memory
                            del tensor, out
                            torch.cuda.empty_cache()

                            done += 1
                            if progress_cb:
                                try:
                                    progress_cb(done, total_patches)
                                except:
                                    pass
                            if done % 100 == 0:
                                print(f"Progress: {done}/{total_patches} patches ({100*done//total_patches}%)")

                    # Write this stripe's mask directly to output TIFF
                    dst.write(stripe_mask, 1, window=window)

    return output_path

# ── COG Conversion ────────────────────────────────────────

def convert_to_cog(input_tif, output_cog):
    subprocess.run([
        "gdal_translate",
        "-of", "COG",
        "-co", "COMPRESS=LZW",
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

def run_classification(merged_tif, model, output_path, num_classes):
    """
    Runs patch-level classification
    Each 64x64 patch gets one class label
    Output is a classified patch grid .tif
    """
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
                pred   = out.argmax(1).item()
                # Fill entire patch with the predicted class
                full_mask[row:row+PATCH_SIZE, col:col+PATCH_SIZE] = pred

    profile.update(count=1, dtype='uint8')
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(full_mask, 1)

    return output_path

def crop_tif(input_tif, output_tif, size=2000):
    """Crop to top-left size x size pixels for quick testing"""
    with rasterio.open(input_tif) as src:
        profile = src.profile.copy()
        data    = src.read(
            window=rasterio.windows.Window(0, 0, size, size)
        )
        profile.update(width=size, height=size)
        with rasterio.open(output_tif, 'w', **profile) as dst:
            dst.write(data)
    print(f"✅ Cropped to {size}x{size}")
    return output_tif