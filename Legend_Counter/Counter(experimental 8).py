"""
detection_annotator.py
========================
Runs your full symbol detection pipeline and saves ONE annotated floor plan image
showing every detection with color coding, scores, and a summary legend.

HOW TO USE
----------
1. Set SYMBOL_PATH, FLOOR_PLAN_PATH, OUTPUT_DIR, GROUND_TRUTH_COUNT below
2. Run: python detection_annotator.py
3. Open: OUTPUT_DIR/annotated_floor_plan.png

COLOR CODING
------------
  GREEN  box  = template matched (score >= threshold) — counted
  RED    box  = blob found but template score too low  — missed
  ORANGE box  = counted but suspiciously high count per crop (possible double-count)

The score printed above each box tells you exactly why a detection passed or failed.
"""

import cv2
import numpy as np
import os
import collections
import time
import multiprocessing
from functools import partial
from skimage import feature

# =============================================================================
#  CONFIGURATION  ← edit these
# =============================================================================

SYMBOL_PATH          = r"D:\AVIS\Manager\output\small_test\symbols\symbol_4.png"
FLOOR_PLAN_PATH      = r"D:\AVIS\Manager\orb_debug\filtered_floor_plan.png"
OUTPUT_DIR           = "annotated_output"

GROUND_TRUTH_COUNT   = 10        # ← SET THIS to the exact known count

# Pipeline settings (keep in sync with symbol_counter.py)
MAX_ASPECT_RATIO             = 3.0
MIN_REGION_AREA              = 20
MAX_REGION_AREA              = None
BLOB_GROUP_DISTANCE_OVERRIDE = 30
CROP_PADDING                 = None
BLOB_DOWNSCALE               = 0.5

TM_THRESHOLD   = 0.55
TM_ANGLES      = list(range(0, 360, 5))
TM_SCALE_STEPS = 25
TM_SCALE_MIN   = 0.5
TM_SCALE_MAX   = 3.0

# Annotated image settings
FONT            = cv2.FONT_HERSHEY_SIMPLEX
BOX_THICKNESS   = 2
LABEL_FONT_SCALE = 0.45
LABEL_THICKNESS  = 1


# =============================================================================
#  PIPELINE (identical logic to symbol_counter.py)
# =============================================================================

def extract_precise_hsv_colors(bgr_img, num_colors=10):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    border_pixels = np.concatenate([bgr_img[0,:], bgr_img[-1,:],
                                    bgr_img[:,0], bgr_img[:,-1]], axis=0)
    flat_border   = border_pixels.reshape(-1, 3)
    binned_border = (flat_border // 8) * 8
    u_colors, cnts = np.unique(binned_border, axis=0, return_counts=True)
    bg_bgr  = u_colors[cnts.argmax()].astype(np.float32)
    diff    = np.linalg.norm(bgr_img.astype(np.float32) - bg_bgr, axis=2)
    fg_mask = diff > 30
    fg_hsv  = hsv[fg_mask]
    if len(fg_hsv) == 0:
        return []
    binned = fg_hsv.copy()
    binned[:, 0] = (fg_hsv[:, 0] // 10) * 10
    binned[:, 1] = (fg_hsv[:, 1] // 30) * 30
    binned[:, 2] = (fg_hsv[:, 2] // 30) * 30
    counts = collections.Counter([tuple(c) for c in binned])
    min_pixels = len(fg_hsv) * 0.02
    return [c for c, n in counts.most_common(num_colors) if n >= min_pixels]


def highlight_hsv_colors(image, target_hsv_bins, h_tol=10, s_tol=40, v_tol=40):
    if not target_hsv_bins:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        return mask
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for (th, ts, tv) in target_hsv_bins:
        th, ts, tv = int(th), int(ts), int(tv)
        if ts < 30:
            mask |= cv2.inRange(hsv,
                np.array([0,   max(0,ts-s_tol), max(0,tv-v_tol)]),
                np.array([180, min(255,ts+s_tol), min(255,tv+v_tol)]))
        else:
            s_low, s_high = max(0,ts-s_tol), min(255,ts+s_tol)
            v_low, v_high = max(0,tv-v_tol), min(255,tv+v_tol)
            h_low, h_high = th-h_tol, th+h_tol
            if h_low < 0:
                mask |= cv2.inRange(hsv, np.array([180+h_low,s_low,v_low]), np.array([180,s_high,v_high]))
                mask |= cv2.inRange(hsv, np.array([0,s_low,v_low]),         np.array([h_high,s_high,v_high]))
            elif h_high > 180:
                mask |= cv2.inRange(hsv, np.array([h_low,s_low,v_low]),     np.array([180,s_high,v_high]))
                mask |= cv2.inRange(hsv, np.array([0,s_low,v_low]),         np.array([h_high-180,s_high,v_high]))
            else:
                mask |= cv2.inRange(hsv, np.array([h_low,s_low,v_low]),     np.array([h_high,s_high,v_high]))
    return mask


def remove_lines_and_noise(binary_img):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    clean = np.zeros_like(binary_img)
    for label in range(1, num_labels):
        rw   = stats[label, cv2.CC_STAT_WIDTH]
        rh   = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        if area < MIN_REGION_AREA:
            continue
        if MAX_REGION_AREA is not None and area > MAX_REGION_AREA:
            continue
        longer  = max(rw, rh)
        shorter = min(rw, rh)
        aspect  = longer / shorter if shorter > 0 else 999
        if aspect > MAX_ASPECT_RATIO:
            continue
        clean[labels == label] = 255
    return clean


def get_symbol_sigma(bgr_img):
    h, w = bgr_img.shape[:2]
    corners   = np.array([bgr_img[0,0], bgr_img[0,-1],
                           bgr_img[-1,0], bgr_img[-1,-1]], dtype=np.float32)
    bg_bgr    = corners.mean(axis=0)
    diff      = np.linalg.norm(bgr_img.astype(np.float32) - bg_bgr, axis=2)
    fg_mask   = (diff > 30).astype(np.uint8) * 255
    coords    = cv2.findNonZero(fg_mask)
    if coords is None:
        fg_w, fg_h = w, h
    else:
        _, _, fg_w, fg_h = cv2.boundingRect(coords)
    diameter  = min(fg_w, fg_h)
    sigma_est = diameter / (2 * 1.414)
    sigma_min = max(1.0, round(sigma_est * 0.6, 1))
    sigma_max = round(sigma_est * 1.4, 1)
    group_dist = int((fg_w**2 + fg_h**2)**0.5 * 0.5)
    return sigma_min, sigma_max, group_dist, fg_w, fg_h


def detect_blobs(filtered_img, sigma_min, sigma_max, threshold=0.05):
    h, w = filtered_img.shape[:2]
    if BLOB_DOWNSCALE < 1.0:
        small = cv2.resize(filtered_img,
                           (int(w*BLOB_DOWNSCALE), int(h*BLOB_DOWNSCALE)),
                           interpolation=cv2.INTER_AREA)
        s_min, s_max = sigma_min*BLOB_DOWNSCALE, sigma_max*BLOB_DOWNSCALE
    else:
        small, s_min, s_max = filtered_img, sigma_min, sigma_max
    blobs = feature.blob_dog(small.astype(np.float64)/255.0,
                             min_sigma=s_min, max_sigma=s_max,
                             sigma_ratio=1.2, threshold=threshold)
    if len(blobs) == 0:
        return []
    blobs[:, 2] *= 1.414
    scale_back = 1.0 / BLOB_DOWNSCALE
    return [(int(b[1]*scale_back), int(b[0]*scale_back), float(b[2]*scale_back))
            for b in blobs]


def group_blobs(blobs, group_distance):
    if not blobs:
        return []
    n = len(blobs)
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        parent[find(i)] = find(j)
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, _ = blobs[i]
            xj, yj, _ = blobs[j]
            if ((xi-xj)**2+(yi-yj)**2)**0.5 <= group_distance:
                union(i, j)
    groups = collections.defaultdict(list)
    for i, blob in enumerate(blobs):
        groups[find(i)].append(blob)
    return list(groups.values())


def crop_groups(floor_bgr, groups, padding):
    fh, fw = floor_bgr.shape[:2]
    results = []
    for group in groups:
        x_min = min(x-int(r) for x,y,r in group)
        y_min = min(y-int(r) for x,y,r in group)
        x_max = max(x+int(r) for x,y,r in group)
        y_max = max(y+int(r) for x,y,r in group)
        x1 = max(0,  x_min - padding)
        y1 = max(0,  y_min - padding)
        x2 = min(fw, x_max + padding)
        y2 = min(fh, y_max + padding)
        crop = floor_bgr[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            continue
        results.append((crop, (x1, y1, x2, y2)))
    return results


def _rotate_template(img, angle):
    h, w = img.shape[:2]
    M    = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    cos, sin = abs(M[0,0]), abs(M[0,1])
    nw = int(h*sin + w*cos)
    nh = int(h*cos + w*sin)
    M[0,2] += nw/2 - w/2
    M[1,2] += nh/2 - h/2
    return cv2.warpAffine(img, M, (nw, nh), borderMode=cv2.BORDER_REPLICATE)


def prepare_templates(symbol_bgr):
    sym_gray = cv2.cvtColor(symbol_bgr, cv2.COLOR_BGR2GRAY)
    sym_h, sym_w = sym_gray.shape[:2]
    scales = np.linspace(TM_SCALE_MIN, TM_SCALE_MAX, TM_SCALE_STEPS)
    templates = []
    for scale in scales:
        sw = max(4, int(sym_w*scale))
        sh = max(4, int(sym_h*scale))
        scaled = cv2.resize(sym_gray, (sw, sh), interpolation=cv2.INTER_AREA)
        for angle in TM_ANGLES:
            rot = _rotate_template(scaled, angle) if angle != 0 else scaled
            templates.append((rot, round(float(scale), 2), angle))
    return templates


def match_template_worker(crop_tuple, templates, threshold):
    crop_bgr, box = crop_tuple
    crop_h, crop_w = crop_bgr.shape[:2]
    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    best_score, best_scale, best_angle = -1.0, 1.0, 0
    best_res_map, best_template = None, None

    for template, scale, angle in templates:
        t_h, t_w = template.shape[:2]
        if t_h > crop_h or t_w > crop_w:
            continue
        res = cv2.matchTemplate(crop_gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score, best_scale, best_angle = max_val, scale, angle
            best_res_map, best_template = res, template

    passed    = best_res_map is not None and best_score >= threshold
    count     = 0
    peak_locs = []

    if passed:
        t_h, t_w         = best_template.shape[:2]
        natural_sym_size = int(max(t_h, t_w) / best_scale)
        suppression_r    = max(natural_sym_size // 2, 4)
        res_map          = best_res_map.copy()
        while True:
            _, max_val, _, max_loc = cv2.minMaxLoc(res_map)
            if max_val < threshold:
                break
            count += 1
            peak_locs.append(max_loc)
            x, y = max_loc
            res_map[max(0,y-suppression_r):min(res_map.shape[0],y+suppression_r),
                    max(0,x-suppression_r):min(res_map.shape[1],x+suppression_r)] = 0.0

    return {
        "count":     count,
        "score":     best_score,
        "scale":     best_scale,
        "angle":     best_angle,
        "passed":    passed,
        "box":       box,
        "crop_bgr":  crop_bgr,
    }


# =============================================================================
#  ANNOTATED IMAGE
# =============================================================================

def _draw_label_with_background(img, text, origin, font, scale, thickness,
                                  text_color, bg_color):
    """Draw text with a filled rectangle behind it for readability."""
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    # background rect
    cv2.rectangle(img, (x, y - th - baseline), (x + tw + 2, y + baseline),
                  bg_color, -1)
    # text
    cv2.putText(img, text, (x+1, y), font, scale, text_color, thickness, cv2.LINE_AA)


def save_annotated_image(floor_bgr, results, output_path):
    """
    Draws all detection boxes on the floor plan with:
      GREEN  = matched (count >= 1)
      RED    = missed  (score below threshold)
      ORANGE = matched but count > 1 per crop (possible double-count)
    Also saves per-crop debug strips to OUTPUT_DIR/crops/
    """
    ann = floor_bgr.copy()
    fh, fw = ann.shape[:2]

    total_count   = 0
    matched_crops = 0
    missed_crops  = 0
    suspect_crops = 0   # count > 1

    # Save individual crop debug strips
    crops_dir = os.path.join(os.path.dirname(output_path), "crops")
    os.makedirs(crops_dir, exist_ok=True)

    for idx, r in enumerate(results):
        x1, y1, x2, y2 = r["box"]
        count  = r["count"]
        score  = r["score"]
        passed = r["passed"]
        total_count += count

        # ── pick color ──────────────────────────────────────────────────
        if not passed:
            box_color = (0, 50, 220)      # RED   — missed
            missed_crops += 1
        elif count > 1:
            box_color = (0, 165, 255)     # ORANGE — suspect double-count
            matched_crops += 1
            suspect_crops += 1
        else:
            box_color = (30, 200, 30)     # GREEN  — clean match
            matched_crops += 1

        # ── draw box ────────────────────────────────────────────────────
        cv2.rectangle(ann, (x1, y1), (x2, y2), box_color, BOX_THICKNESS)

        # ── score label above the box ───────────────────────────────────
        status = "OK" if passed else "MISS"
        label  = f"#{idx+1} {score:.2f} {status} {r['scale']:.1f}x {r['angle']}d"
        label_y = max(y1 - 2, 12)
        _draw_label_with_background(ann, label, (x1, label_y),
                                    FONT, LABEL_FONT_SCALE, LABEL_THICKNESS,
                                    (255, 255, 255), box_color)

        # ── count badge inside box ──────────────────────────────────────
        if count > 0:
            badge = str(count)
            bx    = x1 + 3
            by    = y1 + 18
            if by < y2:
                _draw_label_with_background(ann, badge, (bx, by),
                                            FONT, 0.65, 2,
                                            (255, 255, 255), box_color)

        # ── save crop strip ─────────────────────────────────────────────
        crop = r["crop_bgr"]
        ch, cw = crop.shape[:2]

        # Header bar for the crop strip
        bar_h  = 28
        header = np.zeros((bar_h, cw, 3), dtype=np.uint8)
        header[:] = (50, 50, 50)
        hdr_text = (f"#{idx+1:03d} | {status} | score={score:.4f} | "
                    f"count={count} | scale={r['scale']:.2f}x | angle={r['angle']}d")
        h_color  = (0, 255, 0) if passed else (80, 80, 255)
        cv2.putText(header, hdr_text, (4, 20), FONT, 0.42, h_color, 1, cv2.LINE_AA)
        strip = np.vstack([header, crop])

        crop_path = os.path.join(crops_dir, f"crop_{idx+1:03d}_{'OK' if passed else 'MISS'}.png")
        cv2.imwrite(crop_path, strip)

    # ── summary legend (bottom-left corner) ─────────────────────────────
    legend_lines = [
        f"TOTAL COUNT : {total_count}",
        f"GROUND TRUTH: {GROUND_TRUTH_COUNT}",
        f"ERROR       : {total_count - GROUND_TRUTH_COUNT:+d}",
        f"------",
        f"Matched crops  : {matched_crops}  (green)",
        f"Missed crops   : {missed_crops}  (red)",
        f"Suspect crops  : {suspect_crops}  (orange, count>1)",
        f"Threshold      : {TM_THRESHOLD}",
        f"Blobs->Groups  : {len(results)} crops",
    ]

    line_h     = 22
    legend_w   = 320
    legend_h   = line_h * len(legend_lines) + 14
    legend_x   = 10
    legend_y   = fh - legend_h - 10

    # background
    cv2.rectangle(ann,
                  (legend_x - 4, legend_y - 4),
                  (legend_x + legend_w, legend_y + legend_h),
                  (20, 20, 20), -1)
    cv2.rectangle(ann,
                  (legend_x - 4, legend_y - 4),
                  (legend_x + legend_w, legend_y + legend_h),
                  (180, 180, 180), 1)

    err = total_count - GROUND_TRUTH_COUNT
    for i, line in enumerate(legend_lines):
        color = (200, 200, 200)
        if "TOTAL" in line:
            color = (0, 255, 0) if err == 0 else (80, 80, 255)
        elif "ERROR" in line:
            color = (0, 255, 0) if err == 0 else (0, 100, 255)
        cv2.putText(ann, line,
                    (legend_x, legend_y + 14 + i * line_h),
                    FONT, 0.48, color, 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, ann)
    print(f"\n  Annotated image : {output_path}")
    print(f"  Crop strips     : {crops_dir}/")
    return ann


# =============================================================================
#  MAIN
# =============================================================================

def main():
    t_start = time.perf_counter()

    symbol_bgr = cv2.imread(SYMBOL_PATH, cv2.IMREAD_COLOR)
    if symbol_bgr is None:
        raise FileNotFoundError(f"Cannot open symbol: {SYMBOL_PATH}")
    floor_bgr = cv2.imread(FLOOR_PLAN_PATH, cv2.IMREAD_COLOR)
    if floor_bgr is None:
        raise FileNotFoundError(f"Cannot open floor plan: {FLOOR_PLAN_PATH}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1-2: HSV filter + clean ─────────────────────────────────────
    print("Step 1-2: HSV filter...")
    target_colors = extract_precise_hsv_colors(symbol_bgr)
    filtered      = highlight_hsv_colors(floor_bgr, target_colors)
    filtered      = cv2.dilate(filtered, np.ones((3,3), np.uint8), iterations=1)
    filtered      = remove_lines_and_noise(filtered)

    # Save filtered mask so you can check what the HSV filter kept
    debug_red = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    debug_red[filtered > 0] = (0, 0, 255)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "debug_hsv_mask.png"), debug_red)
    kept_pct = 100 * np.sum(filtered > 0) / (filtered.shape[0] * filtered.shape[1])
    print(f"  HSV mask kept {kept_pct:.2f}% of pixels  (saved: debug_hsv_mask.png)")

    # ── Step 3: blob detection ─────────────────────────────────────────
    print("Step 3: Blob detection...")
    sigma_min, sigma_max, auto_group_dist, sym_fw, sym_fh = get_symbol_sigma(symbol_bgr)
    blobs = detect_blobs(filtered, sigma_min, sigma_max)
    print(f"  Blobs found: {len(blobs)}  (ground truth: {GROUND_TRUTH_COUNT})")
    if len(blobs) == 0:
        print("\n  !! ZERO blobs found — all symbols will be missed.")
        print("     Most likely cause: HSV filter not matching symbol colors in floor plan.")
        print("     Open debug_hsv_mask.png — if it's mostly black, the filter is wrong.")
        return

    # ── Step 4: group + crop ──────────────────────────────────────────
    print("Step 4: Grouping and cropping...")
    group_dist = BLOB_GROUP_DISTANCE_OVERRIDE if BLOB_GROUP_DISTANCE_OVERRIDE else auto_group_dist
    groups     = group_blobs(blobs, group_dist)

    crop_padding = CROP_PADDING if CROP_PADDING else max(int(max(sym_fw, sym_fh) * 0.5), 10)
    crops        = crop_groups(floor_bgr, groups, crop_padding)
    print(f"  Groups: {len(groups)}  |  Crops: {len(crops)}  |  "
          f"Group distance: {group_dist}px  |  Padding: {crop_padding}px")

    # ── Step 5: template matching ─────────────────────────────────────
    print("Step 5: Template matching...")
    templates = prepare_templates(symbol_bgr)
    print(f"  Templates prepared: {len(templates)}")

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        func    = partial(match_template_worker, templates=templates, threshold=TM_THRESHOLD)
        results = pool.map(func, crops)

    # ── Print per-crop table ──────────────────────────────────────────
    print(f"\n  {'#':>4}  {'Score':>7}  {'Pass':>5}  {'Count':>6}  {'Scale':>6}  {'Angle':>6}  Box")
    print(f"  {'─'*4}  {'─'*7}  {'─'*5}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*20}")
    total_count = 0
    for i, r in enumerate(results):
        total_count += r["count"]
        flag = ""
        if not r["passed"] and r["score"] >= 0.45:
            flag = " ← borderline (lower threshold?)"
        elif r["count"] > 1:
            flag = " ← multi-count (check suppression radius)"
        print(f"  {i+1:>4}  {r['score']:>7.4f}  "
              f"{'YES' if r['passed'] else 'NO':>5}  "
              f"{r['count']:>6}  "
              f"{r['scale']:>6.2f}  "
              f"{r['angle']:>6}  "
              f"{r['box']}{flag}")

    err = total_count - GROUND_TRUTH_COUNT
    print(f"\n  ── RESULT ──────────────────────────────────────────")
    print(f"  Total count  : {total_count}")
    print(f"  Ground truth : {GROUND_TRUTH_COUNT}")
    print(f"  Error        : {err:+d}  "
          f"({'CORRECT' if err == 0 else 'OVERCOUNTING' if err > 0 else 'UNDERCOUNTING'})")

    # ── Save annotated image ──────────────────────────────────────────
    print("\nSaving annotated image...")
    out_path = os.path.join(OUTPUT_DIR, "annotated_floor_plan.png")
    save_annotated_image(floor_bgr, results, out_path)

    t_end = time.perf_counter()
    print(f"\n  Done in {t_end - t_start:.1f}s")
    print(f"\n  NEXT STEPS based on error ({err:+d}):")
    if err < 0:
        missed = sum(1 for r in results if not r["passed"])
        borderline = sum(1 for r in results if not r["passed"] and r["score"] >= 0.45)
        print(f"  Undercounting by {abs(err)}.")
        if missed > 0:
            print(f"  → {missed} crops scored below threshold (TM_THRESHOLD={TM_THRESHOLD})")
        if borderline > 0:
            print(f"  → {borderline} crops scored 0.45–0.54 — try lowering TM_THRESHOLD to 0.45")
        if len(blobs) < GROUND_TRUTH_COUNT * 0.8:
            print(f"  → Only {len(blobs)} blobs for {GROUND_TRUTH_COUNT} symbols — "
                  f"check debug_hsv_mask.png, HSV filter may be missing symbols")
    elif err > 0:
        suspect = sum(1 for r in results if r["count"] > 1)
        print(f"  Overcounting by {err}.")
        if suspect > 0:
            print(f"  → {suspect} crops returned count > 1 — suppression radius may be too small")
        if len(blobs) > GROUND_TRUTH_COUNT * 2:
            print(f"  → {len(blobs)} blobs for {GROUND_TRUTH_COUNT} symbols — "
                  f"HSV filter too broad, noise blobs inflating crop count")
    else:
        print(f"  Count is correct!")


if __name__ == "__main__":
    main()