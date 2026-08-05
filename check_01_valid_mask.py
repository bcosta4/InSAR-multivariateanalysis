r"""
check_01_valid_mask.py
───────────────────────
Diagnostic: verifies that 01_prepare_reference_grid.py produced a correctly
oriented valid_mask.tif, i.e. 1 = valid InSAR pixel, 0 = nodata pixel,
matching its own docstring and printed 'Valid pixels' count.

Run this AFTER 01_prepare_reference_grid.py and compare the printed numbers
against the console output that 01 produced (it prints something like:
    "Valid pixels: N / (H x W) = XX.X%").
"""

import numpy as np
import rasterio
from pathlib import Path
import json

OUT_DIR = Path(__file__).parent / "outputs" if (Path(__file__).parent / "outputs").exists() else Path(r"E:\\Processing\\InSAR_multivariate_analysis\\scripts\\outputs")

MASK_PATH = OUT_DIR / "valid_mask.tif"
SUB_PATH  = OUT_DIR / "subsidence_clean.tif"
META_PATH = OUT_DIR / "reference_grid_meta.json"

print("=== Checking 01_prepare_reference_grid.py output ===")

with rasterio.open(MASK_PATH) as m:
    mask = m.read(1)
    mask_nodata = m.nodata

with rasterio.open(SUB_PATH) as s:
    sub = s.read(1)
    sub_nodata = s.nodata

n_total = mask.size
n_ones = int((mask == 1).sum())
n_zeros = int((mask == 0).sum())
n_other = n_total - n_ones - n_zeros

print(f"valid_mask.tif shape        : {mask.shape}")
print(f"valid_mask.tif dtype/nodata : {mask.dtype} / {mask_nodata}")
print(f"Pixels == 1                : {n_ones} ({n_ones/n_total*100:.2f}%)")
print(f"Pixels == 0                : {n_zeros} ({n_zeros/n_total*100:.2f}%)")
print(f"Pixels == other (nodata)   : {n_other} ({n_other/n_total*100:.2f}%)")

# Cross-check: subsidence_clean.tif has NaN wherever InSAR data is invalid,
# and a real float value wherever it's valid. This is the ground truth of
# "valid pixel" independent of how 01 built valid_mask, so we compare directly.
sub_is_valid = ~np.isnan(sub)
sub_is_nodata = np.isnan(sub)

n_valid_in_sub = int(sub_is_valid.sum())
print(f"\\nGround truth from subsidence_clean.tif (non-NaN = truly valid):")
print(f"Truly valid pixels (non-NaN in subsidence_clean.tif): {n_valid_in_sub} "
      f"({n_valid_in_sub/n_total*100:.2f}%)")

# Compare mask==1 against ground truth "valid" set
agree_mask1_is_valid = int(((mask == 1) & sub_is_valid).sum())
agree_mask0_is_nodata = int(((mask == 0) & sub_is_nodata).sum())
mismatch_mask1_but_nodata = int(((mask == 1) & sub_is_nodata).sum())
mismatch_mask0_but_valid = int(((mask == 0) & sub_is_valid).sum())

print(f"\\nCross-check against subsidence_clean.tif:")
print(f"mask==1 AND truly valid     : {agree_mask1_is_valid}")
print(f"mask==0 AND truly nodata    : {agree_mask0_is_nodata}")
print(f"mask==1 BUT truly nodata    : {mismatch_mask1_but_nodata}  <- should be ~0 if mask is correct")
print(f"mask==0 BUT truly valid     : {mismatch_mask0_but_valid}  <- should be ~0 if mask is correct")

if mismatch_mask1_but_nodata > 0.5 * n_valid_in_sub or mismatch_mask0_but_valid > 0.5 * n_valid_in_sub:
    print("\\n[RESULT] *** valid_mask.tif POLARITY LOOKS INVERTED ***")
    print("01_prepare_reference_grid.py likely has a bug: 'validmask = np.isnan(data)...'")
    print("should be 'validmask = ~np.isnan(data)...' (i.e. invert the boolean).")
else:
    print("\\n[RESULT] valid_mask.tif polarity looks CORRECT (1=valid, 0=nodata).")

if META_PATH.exists():
    with open(META_PATH) as f:
        meta = json.load(f)
    print(f"\\nreference_grid_meta.json says nvalid = {meta.get('nvalid')}")
    print(f"Compare this to 'Pixels == 1' above and to 'Truly valid pixels' above -- "
          f"all three numbers should match closely.")
