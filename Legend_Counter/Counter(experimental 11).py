from PIL import Image
import numpy as np
import collections
import cv2
import os

# ════════════════════════════════════════════════════════════════
#  PATHS
# ════════════════════════════════════════════════════════════════
SYMBOL_PATH     = r"D:\AVIS\Manager\output\small_test\symbols\symbol_5.png"
FLOOR_PLAN_PATH = r"D:\AVIS\Manager\output\small_test\43001-AJI-04-DWG-IC-BL1-210004-000\floor_plan.png"

COLOR_TOLERANCE     = 60
TOP_N_COLORS        = 60
SIZE_SMALLER_THRESH = 0.8
SIZE_BIGGER_THRESH  = 5
CROP_PADDING        = 50   # extra pixels added around each contour crop before matching

# ── Hybrid Hu Detector Thresholds ───────────────────────────────
HU_THRESHOLD = 8.0     # Diff score threshold (lower = better match)
MIN_AREA     = 300     # Filters out tiny fragments
MAX_AREA     = 800     # Filters out huge merged blobs
ASPECT_MIN   = 0.55    # Reject lines/blobs with wrong proportions
ASPECT_MAX   = 1.9

# ── Diff Scan Search Space (Multi Angle & Scale) ────────────────
SCAN_ANGLES = range(0, 360, 10)         # Rotate every 10 degrees
SCAN_SCALES = [0.8, 0.9, 1.0, 1.1]      # Multiple scales

# ════════════════════════════════════════════════════════════════
#  OUTPUT DIRECTORIES
# ════════════════════════════════════════════════════════════════
DEBUG_DIR = "./adaptive_debug"
CROPS_DIR = os.path.join(DEBUG_DIR, "Crops")
os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════
#  PART 1 — Filter floor plan to symbol colors
# ════════════════════════════════════════════════════════════════
symbol_pil   = Image.open(SYMBOL_PATH).convert("RGB")
pixels       = list(symbol_pil.getdata())
color_counts = collections.Counter(pixels)
bg_color     = color_counts.most_common(1)[0][0]
sym_colors   = [c for c, _ in color_counts.most_common()
                if c != bg_color][:TOP_N_COLORS]

print(f"Background color   : {bg_color}")
print(f"Symbol colors used : {len(sym_colors)}")

floor  = Image.open(FLOOR_PLAN_PATH).convert("RGB")
fp     = np.array(floor, dtype=np.int32)
result = np.full_like(fp, 255, dtype=np.int32)
for sc in sym_colors:
    match = np.all(np.abs(fp - sc) <= COLOR_TOLERANCE, axis=2)
    result[match] = fp[match]

filtered_out = os.path.join(DEBUG_DIR, "filtered_floor_plan.png")
Image.fromarray(result.astype(np.uint8)).save(filtered_out)
print(f"Saved filtered floor plan -> {filtered_out}")


# ════════════════════════════════════════════════════════════════
#  PART 2 — Contour detection on filtered floor plan
# ════════════════════════════════════════════════════════════════

def build_nonwhite_mask(img, v_thresh=210, s_thresh=25):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    return cv2.bitwise_or(
        (v < v_thresh).astype(np.uint8) * 255,
        (s > s_thresh).astype(np.uint8) * 255)


def suppress_boxes(boxes, iou_thresh=0.3, contain_thresh=0.6):
    if not boxes:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    x1 = boxes[:, 0]; y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]; y2 = boxes[:, 1] + boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = areas.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0., xx2 - xx1) * np.maximum(0., yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        cont  = inter / (np.minimum(areas[i], areas[order[1:]]) + 1e-6)
        order = order[np.where((iou <= iou_thresh) & (cont <= contain_thresh))[0] + 1]
    return [boxes[i].astype(int).tolist() for i in keep]


def get_symbol_bbox(path):
    img = cv2.imread(path)
    msk = build_nonwhite_mask(img)
    msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        h, w = img.shape[:2]; return w, h
    pts = np.concatenate(cnts)
    x, y, w, h = cv2.boundingRect(pts)
    print(f"[INFO] Symbol tight bbox: {w}x{h} px")
    return w, h


sym_w, sym_h = get_symbol_bbox(SYMBOL_PATH)
min_w = sym_w * (1 - SIZE_SMALLER_THRESH)
max_w = sym_w * (1 + SIZE_BIGGER_THRESH)
min_h = sym_h * (1 - SIZE_SMALLER_THRESH)
max_h = sym_h * (1 + SIZE_BIGGER_THRESH)

image = cv2.imread(filtered_out)
H, W  = image.shape[:2]

# Remove very dark background blobs
gray_fp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, dark = cv2.threshold(gray_fp, 30, 255, cv2.THRESH_BINARY_INV)
bg_cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
bg_mask = np.zeros((H, W), np.uint8)
for c in bg_cnts:
    if cv2.contourArea(c) > 10_000:
        cv2.drawContours(bg_mask, [c], -1, 255, cv2.FILLED)
work = image.copy(); work[bg_mask == 255] = 255

mask        = build_nonwhite_mask(work)
mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
mask_final  = cv2.dilate(mask_closed, np.ones((2, 2), np.uint8), iterations=1)

cnts, hier = cv2.findContours(mask_final, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
raw_boxes  = []; rejected = 0
if hier is not None:
    for i, cnt in enumerate(cnts):
        if hier[0][i][3] != -1: continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 15 or h < 15 or cv2.contourArea(cnt) < 150: continue
        if not (min_w <= w <= max_w and min_h <= h <= max_h):
            rejected += 1; continue
        raw_boxes.append([x, y, w, h])

suppressed = suppress_boxes(raw_boxes)
print(f"[INFO] Rejected by size : {rejected}")
print(f"[INFO] Crops to process : {len(suppressed)}")


# ════════════════════════════════════════════════════════════════
#  PART 3 — Symbol preparation  (shared by all detectors)
# ════════════════════════════════════════════════════════════════

def extract_largest_component(msk):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(msk, connectivity=8)
    if n <= 1: return msk
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(msk); out[labels == largest] = 255
    return out


def prepare_symbol(sym_bgr):
    """Remove bg, kill stray blobs, tight-crop. Returns (bgr, mask)."""
    hsv = cv2.cvtColor(sym_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    fg = cv2.bitwise_or((v < 210).astype(np.uint8) * 255,
                        (s >  25).astype(np.uint8) * 255)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    fg = extract_largest_component(fg)
    ys, xs = np.where(fg > 0)
    if ys.size == 0:
        bgr = sym_bgr.copy()
        return bgr, np.ones(bgr.shape[:2], np.uint8) * 255
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    bgr = sym_bgr[y0:y1, x0:x1].copy()
    msk = fg[y0:y1, x0:x1].copy()
    bgr[msk == 0] = 255
    print(f"[INFO] Cleaned symbol  : {bgr.shape[1]}x{bgr.shape[0]} px"
          f"  (was {sym_bgr.shape[1]}x{sym_bgr.shape[0]})")
    return bgr, msk


symbol_bgr_raw       = cv2.imread(SYMBOL_PATH)
sym_bgr, sym_mask    = prepare_symbol(symbol_bgr_raw)
sym_gray             = cv2.cvtColor(sym_bgr, cv2.COLOR_BGR2GRAY)

cv2.imwrite(os.path.join(DEBUG_DIR, "symbol_cleaned.png"),      sym_bgr)
cv2.imwrite(os.path.join(DEBUG_DIR, "symbol_cleaned_mask.png"), sym_mask)


# ════════════════════════════════════════════════════════════════
#  Shared NMS
# ════════════════════════════════════════════════════════════════

def _nms(cands, iou_thresh=0.3):
    if not cands: return []
    arr   = np.array([[c['x'], c['y'], c['w'], c['h'], c['score']]
                      for c in cands], dtype=np.float32)
    x1s   = arr[:, 0]; y1s = arr[:, 1]
    x2s   = arr[:, 0] + arr[:, 2]; y2s = arr[:, 1] + arr[:, 3]
    areas = (x2s - x1s) * (y2s - y1s)
    
    # Sort by score ascending (lower score is better for diffs!)
    order = arr[:, 4].argsort(); kept = []
    while order.size > 0:
        i = order[0]; kept.append(i)
        xx1 = np.maximum(x1s[i], x1s[order[1:]])
        yy1 = np.maximum(y1s[i], y1s[order[1:]])
        xx2 = np.minimum(x2s[i], x2s[order[1:]])
        yy2 = np.minimum(y2s[i], y2s[order[1:]])
        inter = np.maximum(0., xx2 - xx1) * np.maximum(0., yy2 - yy1)
        iou_v = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou_v <= iou_thresh)[0] + 1]
    return [cands[i] for i in kept]


# ════════════════════════════════════════════════════════════════
#  DETECTOR — Hybrid Multi-Angle / Multi-Scale Log-Hu matching
# ════════════════════════════════════════════════════════════════

def match_hu_diff(hu1, hu2):
    """Lower score = better match"""
    return np.mean(np.abs(hu1 - hu2))

def build_hu_scan_variants(sym_mask, angles, scales):
    variants = []
    h, w = sym_mask.shape[:2]
    cx, cy = w / 2., h / 2.
    
    # Pre-compute log-Hu for the base mask
    for ang in angles:
        M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
        ca = abs(M[0, 0]); sa = abs(M[0, 1])
        nw = int(h * sa + w * ca); nh = int(h * ca + w * sa)
        M[0, 2] += nw / 2. - cx; M[1, 2] += nh / 2. - cy
        
        rm = cv2.warpAffine(sym_mask, M, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0)
        for sc in scales:
            sw = max(4, int(nw * sc)); sh = max(4, int(nh * sc))
            sm = cv2.resize(rm, (sw, sh), interpolation=cv2.INTER_NEAREST)
            _, sm = cv2.threshold(sm, 127, 255, cv2.THRESH_BINARY)
            
            cnts, _ = cv2.findContours(sm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts: continue
            main = max(cnts, key=cv2.contourArea)
            moments = cv2.moments(main)
            if moments["m00"] == 0: continue
            
            hu = cv2.HuMoments(moments).flatten()
            hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
            variants.append({'angle': ang, 'scale': sc, 'hu': hu})
            
    print(f"[HU SCAN] Created {len(variants)} scanning variants")
    return variants


def detect_hybrid_hu(crop_bgr, variants_bank, score_thresh):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    
    # Dynamic OTSU thresholding handles local crop intensities well
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Separate touching symbols and erase text loops (like 'O')
    kernel_sep = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_sep)
    
    # Close small gaps inside symbol bodies
    kernel_close = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cands = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0: continue
        aspect = w / float(h)
        
        if area < MIN_AREA or area > MAX_AREA: continue
        if aspect < ASPECT_MIN or aspect > ASPECT_MAX: continue
        
        moments = cv2.moments(cnt)
        if moments["m00"] == 0: continue
        
        hu = cv2.HuMoments(moments).flatten()
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
        
        # Diff scan: find the best match across all generated angles & scales
        best_score = float('inf')
        best_ang = 0
        best_sc = 1.0
        
        for var in variants_bank:
            score = match_hu_diff(var['hu'], hu)
            if score < best_score:
                best_score = score
                best_ang = var['angle']
                best_sc = var['scale']
                
        if best_score < score_thresh:
            cands.append({'x': x, 'y': y, 'w': w, 'h': h,
                          'score': best_score, 'angle': best_ang, 'scale': best_sc,
                          'method': 'HYBRID_HU'})
            
    return _nms(cands, iou_thresh=0.3)


# ════════════════════════════════════════════════════════════════
#  Build diff scan variants bank before loop
# ════════════════════════════════════════════════════════════════
hu_variants = build_hu_scan_variants(sym_mask, SCAN_ANGLES, SCAN_SCALES)


# ════════════════════════════════════════════════════════════════
#  Annotation helper
# ════════════════════════════════════════════════════════════════

COLORS = {'HYBRID_HU': (255, 0, 255)}

def draw_hits(base, hits):
    out = base.copy()
    for h in hits:
        col = COLORS.get(h['method'], (128, 128, 128))
        lbl = f"HU {h['score']:.2f}"
        cv2.rectangle(out, (h['x'], h['y']),
                      (h['x'] + h['w'], h['y'] + h['h']), col, 2)
        cv2.putText(out, lbl, (h['x'], max(h['y'] - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
    return out


# ════════════════════════════════════════════════════════════════
#  PART 4 — Per-crop detection (Hybrid Detector)
# ════════════════════════════════════════════════════════════════

res_hybrid = image.copy()
totals = {'HYBRID_HU': 0}
instance_id = 0

for idx, (bx, by, bw, bh) in enumerate(suppressed):
    x1 = max(0, bx - CROP_PADDING);       y1 = max(0, by - CROP_PADDING)
    x2 = min(W, bx + bw + CROP_PADDING);  y2 = min(H, by + bh + CROP_PADDING)
    crop = image[y1:y2, x1:x2].copy()

    hits = detect_hybrid_hu(crop, hu_variants, HU_THRESHOLD)

    print(f"  [CROP {idx:04d}] ({x1},{y1})->({x2},{y2}) | HYBRID_HU:{len(hits)}")

    crop_hits = draw_hits(crop, hits)
    cv2.imwrite(os.path.join(CROPS_DIR, f"crop_{idx:04d}_HYBRID.png"), crop_hits)

    col = COLORS['HYBRID_HU']
    for hit in hits:
        fx = x1 + hit['x']; fy = y1 + hit['y']
        fw = hit['w'];       fh = hit['h']
        cv2.rectangle(res_hybrid, (fx, fy), (fx + fw, fy + fh), col, 2)
        cv2.putText(res_hybrid, f"{hit['score']:.2f}",
                    (fx, max(fy - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

        ix1 = max(0, fx); iy1 = max(0, fy)
        ix2 = min(W, fx + fw); iy2 = min(H, fy + fh)
        cv2.imwrite(os.path.join(CROPS_DIR,
                    f"inst_{instance_id:04d}_s{hit['score']:.2f}_a{hit['angle']}_sc{hit['scale']}.png"),
                    image[iy1:iy2, ix1:ix2])
        instance_id  += 1
        totals['HYBRID_HU'] += 1


# ════════════════════════════════════════════════════════════════
#  PART 5 — Save all debug outputs
# ════════════════════════════════════════════════════════════════

cv2.imwrite(os.path.join(DEBUG_DIR, "debug_raw_mask.png"),    mask)
cv2.imwrite(os.path.join(DEBUG_DIR, "debug_closed_mask.png"), mask_final)
cv2.imwrite(os.path.join(DEBUG_DIR, "result_HYBRID.png"),     res_hybrid)

print(f"\n{'='*50}")
print(f"  HYBRID HU detections : {totals['HYBRID_HU']}")
print(f"{'='*50}")
print(f"[INFO] Debug dir : {os.path.abspath(DEBUG_DIR)}")
print(f"[INFO] Crops dir : {os.path.abspath(CROPS_DIR)}")

cv2.imshow("Result - HYBRID", res_hybrid)
cv2.waitKey(0)
cv2.destroyAllWindows()