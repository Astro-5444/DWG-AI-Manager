import cv2
import numpy as np

SYMBOL_PATH     = r"D:\AVIS\Manager\output\small_test\symbols\symbol_5.png"
FLOOR_PLAN_PATH = r"D:\AVIS\Manager\output\small_test\43001-AJI-04-DWG-IC-BL1-210004-000\floor_plan.png"

HU_THRESHOLD = 8.0
ASPECT_MIN   = 0.55
ASPECT_MAX   = 1.9

# ── Debug display helper ───────────────────────────────────────────────────────

def show(title, img, max_w=1400, max_h=900):
    """
    Show an image scaled to fit the screen.
    Prints dimensions so you know exactly what you're looking at.
    Press any key to continue to the next step.
    """
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    disp = cv2.resize(img, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)
    label = f"{title}  [{w}x{h}]  (scale={scale:.2f})  — press any key"
    cv2.imshow(label, disp)
    print(f"\n[STEP] {label}")
    cv2.waitKey(0)
    cv2.destroyWindow(label)


def make_rgb(img):
    """Ensure image is BGR for drawing."""
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


# ── Symbol loading & Hu fingerprinting ────────────────────────────────────────

def get_template_area(symbol_path):
    """Get the contour area of the template symbol in pixels."""
    tmpl = cv2.imread(symbol_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(tmpl, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    main = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(main)
    print(f"[DEBUG] Template contour area: {area:.0f} px²")
    print(f"[DEBUG] Template bounding box: {cv2.boundingRect(main)}")
    return area

def get_template_hu_variants(symbol_path):
    # ── STEP 1: raw symbol ────────────────────────────────────────────────
    tmpl_raw = cv2.imread(symbol_path)
    if tmpl_raw is None:
        raise FileNotFoundError(f"Cannot open symbol: {symbol_path}")
    show("STEP 1 | Symbol — raw", tmpl_raw)

    tmpl = cv2.cvtColor(tmpl_raw, cv2.COLOR_BGR2GRAY)

    # ── STEP 2: threshold ─────────────────────────────────────────────────
    _, binary = cv2.threshold(tmpl, 127, 255, cv2.THRESH_BINARY_INV)
    show("STEP 2 | Symbol — BINARY_INV (what the detector sees)", binary)

    # sanity-check: how many white pixels?
    white_px = cv2.countNonZero(binary)
    print(f"  [DEBUG] White pixels after threshold: {white_px}")
    if white_px == 0:
        print("  [WARNING] Binary is ALL BLACK — symbol may be white-on-white or path wrong!")
    elif white_px == binary.size:
        print("  [WARNING] Binary is ALL WHITE — threshold inverted everything!")

    # ── STEP 3: contour on symbol ─────────────────────────────────────────
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dbg = make_rgb(binary)
    cv2.drawContours(dbg, cnts, -1, (0, 255, 0), 1)
    print(f"  [DEBUG] Contours on template: {len(cnts)}")
    for i, c in enumerate(cnts):
        a = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        print(f"    cnt#{i}: area={a:.0f}  size={w}x{h}")
    show("STEP 3 | Symbol — contours (green)", dbg)

    # ── Build rotated Hu variants ─────────────────────────────────────────
    hu_variants = []
    for angle in range(360):
        h_t, w_t = binary.shape
        cx_t, cy_t = w_t / 2, h_t / 2
        M_rot = cv2.getRotationMatrix2D((cx_t, cy_t), -angle, 1.0)
        cos_a, sin_a = abs(M_rot[0, 0]), abs(M_rot[0, 1])
        nw = int(h_t * sin_a + w_t * cos_a)
        nh = int(h_t * cos_a + w_t * sin_a)
        M_rot[0, 2] += nw / 2 - cx_t
        M_rot[1, 2] += nh / 2 - cy_t
        rot = cv2.warpAffine(binary, M_rot, (nw, nh), borderValue=0)
        contours, _ = cv2.findContours(rot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            main = max(contours, key=cv2.contourArea)
            m = cv2.moments(main)
            if m["m00"] > 0:
                hu = cv2.HuMoments(m).flatten()
                hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
                hu_variants.append(hu)

    print(f"  [DEBUG] Hu variants generated: {len(hu_variants)} (out of 360 angles)")

    # ── STEP 4: show a few sample rotations ───────────────────────────────
    sample_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    tiles = []
    for angle in sample_angles:
        h_t, w_t = binary.shape
        cx_t, cy_t = w_t / 2, h_t / 2
        M_rot = cv2.getRotationMatrix2D((cx_t, cy_t), -angle, 1.0)
        cos_a, sin_a = abs(M_rot[0, 0]), abs(M_rot[0, 1])
        nw = int(h_t * sin_a + w_t * cos_a)
        nh = int(h_t * cos_a + w_t * sin_a)
        M_rot[0, 2] += nw / 2 - cx_t
        M_rot[1, 2] += nh / 2 - cy_t
        rot = cv2.warpAffine(binary, M_rot, (nw, nh), borderValue=0)
        # Pad to uniform size for tiling
        pad_size = 80
        rot_small = cv2.resize(rot, (pad_size, pad_size))
        tile = make_rgb(rot_small)
        cv2.putText(tile, f"{angle}d", (2, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)
        tiles.append(tile)
    combined = np.hstack(tiles)
    show("STEP 4 | Symbol — 8 sample rotations (0/45/90/...)", combined)

    return hu_variants


# ── Floor plan processing ──────────────────────────────────────────────────────

def get_floor_plan_contours(floor_plan_path):
    # ── STEP 5: raw floor plan ────────────────────────────────────────────
    img = cv2.imread(floor_plan_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open floor plan: {floor_plan_path}")
    show("STEP 5 | Floor plan — raw", img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show("STEP 6 | Floor plan — grayscale", gray)

    # ── STEP 7: histogram (print, don't show) ─────────────────────────────
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    dark_px  = int(hist[:64].sum())
    mid_px   = int(hist[64:192].sum())
    light_px = int(hist[192:].sum())
    total    = gray.size
    print(f"\n  [DEBUG] Grayscale histogram buckets:")
    print(f"    Dark  (0-63):   {dark_px:,}  ({100*dark_px/total:.1f}%)")
    print(f"    Mid  (64-191):  {mid_px:,}  ({100*mid_px/total:.1f}%)")
    print(f"    Light(192-255): {light_px:,}  ({100*light_px/total:.1f}%)")
    print(f"  Otsu threshold will fire on the bimodal split — expect it ~50-100 for B&W plans")

    # ── STEP 8: Otsu threshold ────────────────────────────────────────────
    otsu_val, binary = cv2.threshold(gray, 0, 255,
                                     cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print(f"  [DEBUG] Otsu picked threshold = {otsu_val:.1f}")
    show("STEP 7 | Floor plan — after Otsu BINARY_INV", binary)

    # ── STEP 9: morphological OPEN (separates + removes text) ────────────
    kernel_sep = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_sep)
    # Highlight what was removed
    removed_by_open = cv2.subtract(binary, opened)
    dbg_open = make_rgb(opened)
    removed_overlay = np.zeros_like(dbg_open)
    removed_overlay[removed_by_open > 0] = (0, 0, 255)   # RED = destroyed by open
    dbg_open = cv2.addWeighted(dbg_open, 1.0, removed_overlay, 0.7, 0)
    show("STEP 8 | After OPEN 5x5 (RED = pixels removed by open — check if symbols are hurt!)", dbg_open)

    # ── STEP 10: morphological CLOSE (fills gaps) ─────────────────────────
    kernel_close = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
    added_by_close = cv2.subtract(closed, opened)
    dbg_close = make_rgb(closed)
    added_overlay = np.zeros_like(dbg_close)
    added_overlay[added_by_close > 0] = (0, 255, 0)      # GREEN = filled by close
    dbg_close = cv2.addWeighted(dbg_close, 1.0, added_overlay, 0.7, 0)
    show("STEP 9 | After CLOSE 3x3 (GREEN = pixels added by close)", dbg_close)

    binary_final = closed

    # ── STEP 11: all contours ─────────────────────────────────────────────
    contours, _ = cv2.findContours(binary_final, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    print(f"\n  [DEBUG] Total external contours: {len(contours)}")

    # Draw ALL contours colour-coded by area
    dbg_all = make_rgb(binary_final)
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if a < MIN_AREA:
            color = (80, 80, 80)        # dark gray  — too small
        elif a > MAX_AREA:
            color = (0, 0, 200)         # blue       — too large
        else:
            color = (0, 200, 255)       # yellow     — in area range
        cv2.drawContours(dbg_all, [cnt], -1, color, 1)

    # Legend
    legend_y = 20
    for label, col in [("Too small", (80,80,80)),
                        ("Too large", (0,0,200)),
                        ("In area range", (0,200,255))]:
        cv2.rectangle(dbg_all, (10, legend_y-12), (24, legend_y+2), col, -1)
        cv2.putText(dbg_all, label, (28, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        legend_y += 20
    show("STEP 10 | All contours (gray=small  blue=large  yellow=area-OK)", dbg_all)

    return img, binary_final, contours


# ── Match helpers ──────────────────────────────────────────────────────────────

def match_hu(hu1, hu2):
    return np.mean(np.abs(hu1 - hu2))


# ── Main ───────────────────────────────────────────────────────────────────────

template_area = get_template_area(SYMBOL_PATH)

# Allow ±40% size variation around the template
AREA_TOLERANCE = 0.40
MIN_AREA = template_area * (1 - AREA_TOLERANCE)
MAX_AREA = template_area * (1 + AREA_TOLERANCE)

print(f"[DEBUG] Area window: {MIN_AREA:.0f} — {MAX_AREA:.0f} px²")

template_hu_variants = get_template_hu_variants(SYMBOL_PATH)
img, binary_final, contours = get_floor_plan_contours(FLOOR_PLAN_PATH)

matches        = []
rejected_area  = 0
rejected_aspect = 0
rejected_hu    = 0
passed_all     = 0

# Debug image — show per-filter rejections
dbg_filter = img.copy()

for cnt in contours:
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect = w / h if h > 0 else 0

    # ── area filter ───────────────────────────────────────────────────────
    if area < MIN_AREA or area > MAX_AREA:
        rejected_area += 1
        continue

    # ── aspect ratio filter ───────────────────────────────────────────────
    if aspect < ASPECT_MIN or aspect > ASPECT_MAX:
        rejected_aspect += 1
        cv2.rectangle(dbg_filter, (x, y), (x+w, y+h), (0, 100, 255), 1)  # orange
        cv2.putText(dbg_filter, f"ar={aspect:.2f}", (x, y-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 100, 255), 1)
        continue

    # ── Hu moments ───────────────────────────────────────────────────────
    moments = cv2.moments(cnt)
    if moments["m00"] == 0:
        continue

    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    score = min((match_hu(hu_v, hu) for hu_v in template_hu_variants),
                default=float('inf'))

    print(f"  [CANDIDATE] area={area:.0f}  aspect={aspect:.2f}  "
          f"size={w}x{h}  Hu_score={score:.3f}")

    if score >= HU_THRESHOLD:
        rejected_hu += 1
        cv2.rectangle(dbg_filter, (x, y), (x+w, y+h), (200, 0, 200), 1)  # magenta
        cv2.putText(dbg_filter, f"hu={score:.2f}", (x, y-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 0, 200), 1)
        continue

    passed_all += 1
    matches.append((score, x, y, w, h))
    cv2.rectangle(dbg_filter, (x, y), (x+w, y+h), (0, 255, 0), 2)       # green
    cv2.putText(dbg_filter, f"{score:.2f}", (x, y-3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

# ── STEP 12: filter breakdown ──────────────────────────────────────────────
print(f"\n[FILTER SUMMARY]")
print(f"  Total contours:       {len(contours)}")
print(f"  Rejected by area:     {rejected_area}")
print(f"  Rejected by aspect:   {rejected_aspect}  (orange boxes)")
print(f"  Rejected by Hu score: {rejected_hu}  (magenta boxes)")
print(f"  PASSED (matches):     {passed_all}  (green boxes)")

# Legend on filter debug image
ly = 20
for label, col in [("aspect reject", (0,100,255)),
                    ("Hu reject", (200,0,200)),
                    ("MATCH", (0,255,0))]:
    cv2.rectangle(dbg_filter, (10, ly-12), (24, ly+2), col, -1)
    cv2.putText(dbg_filter, label, (28, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    ly += 20

show("STEP 11 | Per-filter rejection map (orange=aspect  magenta=Hu  green=match)", dbg_filter)

# ── STEP 13: final result ──────────────────────────────────────────────────
result = img.copy()
matches.sort(key=lambda m: m[0])
for score, x, y, w, h in matches:
    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(result, f"{score:.2f}", (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

print(f"\n[RESULT] Detected {len(matches)} instances")
for m in matches:
    print(f"  score={m[0]:.3f}  @ ({m[1]},{m[2]})  size={m[3]}x{m[4]}")

show(f"STEP 12 | FINAL — {len(matches)} matches", result)
cv2.destroyAllWindows()