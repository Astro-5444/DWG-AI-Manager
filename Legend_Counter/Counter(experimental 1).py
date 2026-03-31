import cv2
import numpy as np



SYMBOL_PATH     = r"D:\AVIS\Manager\output\DWG\symbols\004 - Copy.png"
FLOOR_PLAN_PATH = r"D:\AVIS\Manager\output\DWG\GROUND FLOOR\filtered_floor_plan.png"

# ──────────────────────────────────────────────
#  Size threshold settings
#  The detected box must be within these ratios
#  of the symbol's bounding box size.
#
#  Example with defaults:
#    symbol is 100x80 px
#    allowed W range: 60px  → 130px   (−40% / +30%)
#    allowed H range: 48px  → 104px   (−40% / +30%)
# ──────────────────────────────────────────────
SIZE_SMALLER_THRESH = 0.8   # reject if smaller than symbol by more than 40%
SIZE_BIGGER_THRESH  = 5  # reject if bigger  than symbol by more than 30%


# ──────────────────────────────────────────────
#  NMS + containment suppression
# ──────────────────────────────────────────────
def suppress_boxes(boxes, iou_thresh=0.3, contain_thresh=0.6):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    x1 = boxes[:, 0];  y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = areas.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w   = np.maximum(0.0, xx2 - xx1)
        h   = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        iou          = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-6)
        smaller_area = np.minimum(areas[i], areas[order[1:]])
        containment  = intersection / (smaller_area + 1e-6)
        keep_mask    = (iou <= iou_thresh) & (containment <= contain_thresh)
        order        = order[np.where(keep_mask)[0] + 1]
    return [boxes[i].astype(int).tolist() for i in keep]


# ──────────────────────────────────────────────
#  Non-white mask builder
# ──────────────────────────────────────────────
def build_nonwhite_mask(img, brightness_thresh=210, saturation_thresh=25):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s_ch, v_ch = cv2.split(hsv)
    dark_mask    = (v_ch < brightness_thresh).astype(np.uint8) * 255
    colored_mask = (s_ch > saturation_thresh).astype(np.uint8) * 255
    return cv2.bitwise_or(dark_mask, colored_mask)


# ──────────────────────────────────────────────
#  Get the tight bounding box of a symbol image
#  (ignores white/background padding around it)
# ──────────────────────────────────────────────
def get_symbol_size(symbol_path):
    sym = cv2.imread(symbol_path)
    if sym is None:
        raise FileNotFoundError(f"Cannot open symbol: {symbol_path}")

    # Build non-white mask on the symbol itself
    sym_mask = build_nonwhite_mask(sym)

    # Close small gaps so the symbol reads as one blob
    close_k  = np.ones((5, 5), np.uint8)
    sym_mask  = cv2.morphologyEx(sym_mask, cv2.MORPH_CLOSE, close_k)

    contours, _ = cv2.findContours(sym_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fall back to full image size if no contour found
        h, w = sym.shape[:2]
        print(f"[WARN] No contour found in symbol — using full image size: {w}x{h}")
        return w, h

    # Use the bounding box that wraps ALL symbol contours together
    all_pts  = np.concatenate(contours, axis=0)
    x, y, w, h = cv2.boundingRect(all_pts)
    print(f"[INFO] Symbol tight bounding box: {w}x{h} px  (at {x},{y})")
    return w, h


# ──────────────────────────────────────────────
#  STEP 0 — Measure the reference symbol
# ──────────────────────────────────────────────
sym_w, sym_h = get_symbol_size(SYMBOL_PATH)

# Compute allowed W and H ranges
min_w = sym_w * (1.0 - SIZE_SMALLER_THRESH)   # e.g. 60% of symbol width
max_w = sym_w * (1.0 + SIZE_BIGGER_THRESH)    # e.g. 130% of symbol width
min_h = sym_h * (1.0 - SIZE_SMALLER_THRESH)
max_h = sym_h * (1.0 + SIZE_BIGGER_THRESH)

print(f"[INFO] Allowed W: {min_w:.0f} – {max_w:.0f} px")
print(f"[INFO] Allowed H: {min_h:.0f} – {max_h:.0f} px")


# ──────────────────────────────────────────────
#  STEP 1 — Load floor plan + remove dark background
# ──────────────────────────────────────────────
image = cv2.imread(FLOOR_PLAN_PATH)
if image is None:
    raise FileNotFoundError(f"Cannot open floor plan: {FLOOR_PLAN_PATH}")

H, W = image.shape[:2]

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, dark_bg_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
bg_contours, _  = cv2.findContours(dark_bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
bg_mask = np.zeros((H, W), dtype=np.uint8)
for cnt in bg_contours:
    if cv2.contourArea(cnt) > 10_000:
        cv2.drawContours(bg_mask, [cnt], -1, 255, thickness=cv2.FILLED)

work = image.copy()
work[bg_mask == 255] = 255


# ──────────────────────────────────────────────
#  STEP 2 — Non-white mask
# ──────────────────────────────────────────────
mask = build_nonwhite_mask(work)


# ──────────────────────────────────────────────
#  STEP 3 — Close gaps + gentle dilate
# ──────────────────────────────────────────────
close_kernel  = np.ones((5, 5), np.uint8)
mask_closed   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
dilate_kernel = np.ones((2, 2), np.uint8)
mask_final    = cv2.dilate(mask_closed, dilate_kernel, iterations=1)


# ──────────────────────────────────────────────
#  STEP 4 — Contours (outer only)
# ──────────────────────────────────────────────
contours, hierarchy = cv2.findContours(mask_final, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

raw_boxes      = []
rejected_size  = 0

if hierarchy is not None:
    hier = hierarchy[0]
    for i, cnt in enumerate(contours):
        if hier[i][3] != -1:
            continue                          # skip hole contours

        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        # Hard minimums (noise guard)
        if w < 15 or h < 15 or area < 150:
            continue

        # ── Symbol-relative size filter ──────────────
        if w < min_w or w > max_w or h < min_h or h > max_h:
            rejected_size += 1
            continue
        # ─────────────────────────────────────────────

        raw_boxes.append([x, y, w, h])


# ──────────────────────────────────────────────
#  STEP 5 — Suppress overlapping / contained boxes
# ──────────────────────────────────────────────
final_boxes = suppress_boxes(raw_boxes, iou_thresh=0.3, contain_thresh=0.6)

print(f"[INFO] Rejected by size filter : {rejected_size}")
print(f"[INFO] Detected after suppression: {len(final_boxes)}")

result = image.copy()
for (x, y, w, h) in final_boxes:
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imwrite("debug_raw_mask.png",    mask)
cv2.imwrite("debug_closed_mask.png", mask_final)
cv2.imwrite("detected_shapes.png",   result)

cv2.imshow("Raw Mask",         mask)
cv2.imshow("Closed Mask",      mask_final)
cv2.imshow("Detection Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()