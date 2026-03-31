from PIL import Image
import numpy as np
import collections


SYMBOL_PATH     = r"D:\AVIS\Manager\output\DWG\symbols\004 - Copy.png"
FLOOR_PLAN_PATH = r"D:\AVIS\Manager\output\DWG\GROUND FLOOR\floor_plan.png"

OUTPUT_PATH     = r"D:\AVIS\Manager\output\DWG\GROUND FLOOR\filtered_floor_plan.png"

COLOR_TOLERANCE = 60  # adjust if needed
TOP_N_COLORS = 60



# ── Step 1: Load symbol ─────────────────────────────────────────────
symbol = Image.open(SYMBOL_PATH).convert("RGB")
pixels = list(symbol.getdata())

# Count colors
color_counts = collections.Counter(pixels)

# Get background (most common color)
background_color = color_counts.most_common(1)[0][0]

# Get symbol colors (everything except background)
# Sort colors by frequency
sorted_colors = color_counts.most_common()

# Remove background
sorted_colors = [c for c in sorted_colors if c[0] != background_color]

# Take top N colors
top_colors = sorted_colors[:TOP_N_COLORS]

# Extract only RGB values
symbol_colors = [c[0] for c in top_colors]


print(f"Background color: {background_color}")
print(f"Symbol colors count: {len(symbol_colors)}")


# ── Step 2: Load floor plan ─────────────────────────────────────────
floor = Image.open(FLOOR_PLAN_PATH).convert("RGB")
fp = np.array(floor, dtype=np.int32)

h, w = fp.shape[:2]

# Output image (start fully background)
result = np.full_like(fp, [255, 255, 255], dtype=np.int32)



# ── Step 3: Keep only symbol colors ─────────────────────────────────
for sym_color in symbol_colors:
    diff = np.abs(fp - sym_color)
    match = np.all(diff <= COLOR_TOLERANCE, axis=2)

    result[match] = fp[match]  # keep original symbol pixel


# ── Step 4: Save result ─────────────────────────────────────────────
Image.fromarray(result.astype(np.uint8)).save(OUTPUT_PATH)

print(f"Saved → {OUTPUT_PATH}")
