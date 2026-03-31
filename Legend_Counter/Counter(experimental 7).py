import cv2 as cv
import numpy as np
import os

# ============================================================
#  CONFIGURATION
# ============================================================
SYMBOL_PATH     = r"D:\AVIS\Manager\output\small_test\symbols\symbol_4.png"
FLOOR_PLAN_PATH = r"D:\AVIS\Manager\output\small_test\43001-AJI-04-DWG-IC-BL1-210004-000\floor_plan.png"

HU_THRESHOLD    = 0.80    # was 0.35
MIN_AREA_RATIO  = 0.3     # contour must be at least 30% of template area
MAX_AREA_RATIO  = 3.0     # contour must be at most 3x template area
NMS_IOU         = 0.30

ANGLES          = [0, 90, 180, 270]
SCALES          = [0.75, 0.85, 1.0, 1.15, 1.30]

DEBUG_DIR       = "orb_debug"   # same folder as before
# ============================================================

os.makedirs(DEBUG_DIR, exist_ok=True)

def save(name: str, img: np.ndarray):
    path = os.path.join(DEBUG_DIR, name)
    cv.imwrite(path, img)
    print(f"  💾 saved → {path}")


def auto_crop(gray: np.ndarray, pad: int = 2) -> np.ndarray:
    _, binary = cv.threshold(gray, 240, 255, cv.THRESH_BINARY_INV)
    coords = cv.findNonZero(binary)
    if coords is None:
        return gray
    x, y, w, h = cv.boundingRect(coords)
    x = max(0, x - pad);  y = max(0, y - pad)
    w = min(gray.shape[1] - x, w + 2 * pad)
    h = min(gray.shape[0] - y, h + 2 * pad)
    return gray[y:y+h, x:x+w]


def iou(a, b) -> float:
    ax1, ay1, aw, ah = a;  ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = b;  bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def nms(detections: list, iou_thresh: float) -> list:
    detections = sorted(detections, key=lambda d: d[1], reverse=True)
    kept = []
    for det in detections:
        if all(iou(det[0], k[0]) < iou_thresh for k in kept):
            kept.append(det)
    return kept


def get_hu(contour) -> np.ndarray:
    """Return log-scaled Hu moments for a contour."""
    moments = cv.moments(contour)
    hu = cv.HuMoments(moments).flatten()
    # Log scale — standard practice to normalize the huge range
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu


def hu_distance(hu1: np.ndarray, hu2: np.ndarray) -> float:
    """L2 distance between two log-Hu moment vectors (lower = more similar)."""
    return float(np.linalg.norm(hu1 - hu2))


# ══════════════════════════════════════════════════════════════
#  STAGE 0 — Load raw inputs
# ══════════════════════════════════════════════════════════════
print("\n[STAGE 0] Loading images …")
img_color = cv.imread(FLOOR_PLAN_PATH, cv.IMREAD_COLOR)
img_tmpl  = cv.imread(SYMBOL_PATH,     cv.IMREAD_COLOR)
assert img_color is not None, f"Cannot load: {FLOOR_PLAN_PATH}"
assert img_tmpl  is not None, f"Cannot load: {SYMBOL_PATH}"

floor_gray    = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
template_gray = cv.cvtColor(img_tmpl,  cv.COLOR_BGR2GRAY)

save("hu_00a_floor_plan_input.png",       img_color)
save("hu_00b_symbol_template_input.png",  img_tmpl)


# ══════════════════════════════════════════════════════════════
#  STAGE 1 — Auto-crop + binarise template
# ══════════════════════════════════════════════════════════════
print("\n[STAGE 1] Crop + binarise template …")
template_crop = auto_crop(template_gray)
# THRESH_BINARY_INV captures the white hole inside
_, tmpl_bin   = cv.threshold(template_crop, 127, 255, cv.THRESH_BINARY_INV)

# Visualise side-by-side: gray crop | binary
h = max(template_crop.shape[0], tmpl_bin.shape[0])
pad_g = np.full((h, template_crop.shape[1]), 255, np.uint8)
pad_b = np.full((h, tmpl_bin.shape[1]),      255, np.uint8)
pad_g[:template_crop.shape[0], :] = template_crop
pad_b[:tmpl_bin.shape[0],      :] = tmpl_bin
divider = np.full((h, 4), 128, np.uint8)
comparison = np.hstack([pad_g, divider, pad_b])
cv.putText(comparison, "GRAY CROP", (2, 12), cv.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
cv.putText(comparison, "BINARY (INV)", (template_crop.shape[1]+6, 12),
           cv.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
# Upscale for visibility
scale_up = max(1, 300 // max(comparison.shape[:2]))
if scale_up > 1:
    comparison = cv.resize(comparison,
                           (comparison.shape[1]*scale_up, comparison.shape[0]*scale_up),
                           interpolation=cv.INTER_NEAREST)
save("hu_01_template_crop_binary.png", comparison)


# ══════════════════════════════════════════════════════════════
#  STAGE 2 — Extract template contour + Hu moments
# ══════════════════════════════════════════════════════════════
print("\n[STAGE 2] Template contour + Hu moments …")
# RETR_LIST gets all contours, we filter out the full-frame one
contours_t, _ = cv.findContours(tmpl_bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
frame_area = template_crop.shape[0] * template_crop.shape[1]
contours_t = [c for c in contours_t
              if cv.contourArea(c) < frame_area * 0.9]  # exclude full-image border
contours_t = sorted(contours_t, key=cv.contourArea, reverse=True)

assert len(contours_t) > 0, "No contours found in template!"
tmpl_contour  = contours_t[0]
tmpl_area     = cv.contourArea(tmpl_contour)
tmpl_hu       = get_hu(tmpl_contour)

print(f"  Template contour area : {tmpl_area:.0f} px²")
print(f"  Template Hu moments   : {tmpl_hu}")

# Draw contour on upscaled template
tmpl_vis = cv.cvtColor(template_crop, cv.COLOR_GRAY2BGR)
cv.drawContours(tmpl_vis, contours_t, 0, (0, 200, 255), 1)
# Mark centroid
M = cv.moments(tmpl_contour)
if M["m00"] > 0:
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    cv.circle(tmpl_vis, (cx, cy), 2, (0, 0, 255), -1)

scale_up = max(1, 300 // max(tmpl_vis.shape[:2]))
if scale_up > 1:
    tmpl_vis = cv.resize(tmpl_vis,
                         (tmpl_vis.shape[1]*scale_up, tmpl_vis.shape[0]*scale_up),
                         interpolation=cv.INTER_NEAREST)

# Add Hu values as text below
info = np.full((90, tmpl_vis.shape[1], 3), 30, np.uint8)
for i, v in enumerate(tmpl_hu[:4]):
    cv.putText(info, f"Hu[{i}]: {v:.3f}", (4, 18 + i*18),
               cv.FONT_HERSHEY_SIMPLEX, 0.42, (180, 255, 180), 1)
tmpl_vis = np.vstack([tmpl_vis, info])
save("hu_02_template_contour_hu.png", tmpl_vis)


# ══════════════════════════════════════════════════════════════
#  STAGE 3 — Floor plan binarisation + all contours
# ══════════════════════════════════════════════════════════════
print("\n[STAGE 3] Floor plan binarisation + contour extraction …")
_, floor_bin = cv.threshold(floor_gray, 127, 255, cv.THRESH_BINARY_INV)
contours_f, _ = cv.findContours(floor_bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
floor_area = floor_gray.shape[0] * floor_gray.shape[1]
contours_f = [c for c in contours_f
              if cv.contourArea(c) < floor_area * 0.9]
print(f"  Total contours on floor plan: {len(contours_f)}")

floor_all_contours = img_color.copy()
cv.drawContours(floor_all_contours, contours_f, -1, (0, 180, 255), 1)
cv.putText(floor_all_contours,
           f"ALL contours: {len(contours_f)}",
           (10, 35), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 180, 255), 2)
save("hu_03_floor_all_contours.png", floor_all_contours)


# ══════════════════════════════════════════════════════════════
#  STAGE 4 — Area filter (removes clearly wrong sizes)
# ══════════════════════════════════════════════════════════════
print("\n[STAGE 4] Area filter …")
area_min = tmpl_area * MIN_AREA_RATIO
area_max = tmpl_area * MAX_AREA_RATIO

area_passed = [c for c in contours_f
               if area_min <= cv.contourArea(c) <= area_max]
print(f"  After area filter [{area_min:.0f}–{area_max:.0f} px²]: "
      f"{len(area_passed)} / {len(contours_f)} contours remain")

floor_area_vis = img_color.copy()
cv.drawContours(floor_area_vis, area_passed, -1, (0, 255, 120), 1)
cv.putText(floor_area_vis,
           f"After area filter: {len(area_passed)} contours  "
           f"[{area_min:.0f}–{area_max:.0f} px2]",
           (10, 35), cv.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 120), 2)
save("hu_04_floor_area_filtered_contours.png", floor_area_vis)


# ══════════════════════════════════════════════════════════════
#  STAGE 5 — Multi-angle Hu matching + score heatmap
# ══════════════════════════════════════════════════════════════
print("\n[STAGE 5] Hu moment matching (multi-angle template Hu) …")

# Pre-compute Hu for all rotations of the template contour
# (Hu moments are rotation-invariant for most, but let's be safe with all angles)
tmpl_hu_variants = []
for angle in ANGLES:
    h_t, w_t = template_crop.shape
    cx_t, cy_t = w_t / 2, h_t / 2
    M_rot = cv.getRotationMatrix2D((cx_t, cy_t), -angle, 1.0)
    cos_a, sin_a = abs(M_rot[0,0]), abs(M_rot[0,1])
    nw = int(h_t * sin_a + w_t * cos_a)
    nh = int(h_t * cos_a + w_t * sin_a)
    M_rot[0, 2] += nw/2 - cx_t
    M_rot[1, 2] += nh/2 - cy_t
    rot_tmpl = cv.warpAffine(tmpl_bin, M_rot, (nw, nh), borderValue=0)
    ctrs, _ = cv.findContours(rot_tmpl, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if ctrs:
        biggest = max(ctrs, key=cv.contourArea)
        tmpl_hu_variants.append(get_hu(biggest))

print(f"  Template Hu variants (one per angle): {len(tmpl_hu_variants)}")

raw = []
score_map = []  # (contour, distance) for visualisation

for cnt in area_passed:
    hu_c  = get_hu(cnt)
    # Best distance across all angle variants
    best_dist = min(hu_distance(hu_c, hu_v) for hu_v in tmpl_hu_variants)
    score_map.append((cnt, best_dist))
    if best_dist <= HU_THRESHOLD:
        x, y, w, h = cv.boundingRect(cnt)
        raw.append(((x, y, w, h), best_dist))

print(f"  Candidates passing Hu threshold ({HU_THRESHOLD}): {len(raw)}")

# Colour-code all area-passed contours by their Hu distance
hu_score_vis = img_color.copy()
for cnt, dist in score_map:
    # Green (close match) → Red (bad match), capped at threshold*2
    t = min(dist / (HU_THRESHOLD * 2), 1.0)
    color = (0, int(255 * (1 - t)), int(255 * t))   # BGR: green→red
    cv.drawContours(hu_score_vis, [cnt], -1, color, 1)

# Legend
cv.putText(hu_score_vis, "Hu distance: GREEN=close  RED=far",
           (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
cv.putText(hu_score_vis, f"Threshold: {HU_THRESHOLD}  |  Candidates: {len(raw)}",
           (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)
save("hu_05_hu_score_heatmap.png", hu_score_vis)


# ══════════════════════════════════════════════════════════════
#  STAGE 6 — Raw detections (before NMS)
# ══════════════════════════════════════════════════════════════
print("\n[STAGE 6] Raw detections before NMS …")
raw_vis = img_color.copy()
for (x, y, w, h), dist in raw:
    cv.rectangle(raw_vis, (x, y), (x+w, y+h), (0, 100, 255), 1)
cv.putText(raw_vis, f"RAW detections: {len(raw)}  (before NMS)",
           (10, 35), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 255), 2)
save("hu_06_raw_detections_before_NMS.png", raw_vis)


# ══════════════════════════════════════════════════════════════
#  STAGE 7 — NMS + final result
# ══════════════════════════════════════════════════════════════
print("\n[STAGE 7] Applying NMS …")

# For NMS we want HIGHER score = better, but dist is LOWER = better → invert
raw_inverted = [(bbox, 1.0 - dist) for bbox, dist in raw]
final        = nms(raw_inverted, NMS_IOU)

final_vis = img_color.copy()
for (x, y, w, h), score in final:
    cv.rectangle(final_vis, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv.putText(final_vis, f"{1-score:.2f}", (x, max(0, y-4)),
               cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
cv.putText(final_vis, f"FINAL count: {len(final)}  (after NMS IoU={NMS_IOU})",
           (10, 45), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
save("hu_07_final_result_after_NMS.png", final_vis)


# ══════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════
print(f"""
{'='*50}
  SYMBOL COUNT : {len(final)}
  Debug images : {DEBUG_DIR}/
{'='*50}
Stages saved:
  hu_00a  floor_plan_input
  hu_00b  symbol_template_input
  hu_01   template_crop_binary       ← what binarisation looks like
  hu_02   template_contour_hu        ← the contour + Hu values
  hu_03   floor_all_contours         ← every contour detected on floor
  hu_04   floor_area_filtered        ← after area size filter
  hu_05   hu_score_heatmap           ← green=similar shape, red=different
  hu_06   raw_detections_before_NMS  ← candidates passing Hu threshold
  hu_07   final_result_after_NMS     ← final answer
""")

cv.imshow("Final Result", final_vis)
cv.waitKey(0)
cv.destroyAllWindows()