import cv2
import numpy as np


# ──────────────────────────────────────────────
#  Improved box suppression:
#  Combines IoU-based NMS  +  containment merging.
#
#  Two boxes are merged/suppressed when:
#    1. IoU  > iou_thresh        (standard overlap)
#    2. The smaller box is contained > contain_thresh
#       inside the larger one   (one wraps the other)
# ──────────────────────────────────────────────
def suppress_boxes(boxes, iou_thresh=0.3, contain_thresh=0.6):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype=np.float32)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    # Process largest boxes first so they absorb smaller ones
    order = areas.argsort()[::-1]
    keep  = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        # Intersection with every remaining box
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w   = np.maximum(0.0, xx2 - xx1)
        h   = np.maximum(0.0, yy2 - yy1)
        intersection = w * h

        # Standard IoU
        iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-6)

        # Containment ratio: what fraction of the SMALLER box is inside box i
        smaller_area = np.minimum(areas[i], areas[order[1:]])
        containment  = intersection / (smaller_area + 1e-6)

        # Keep a box only if it is neither heavily overlapping nor contained
        keep_mask = (iou <= iou_thresh) & (containment <= contain_thresh)
        order = order[np.where(keep_mask)[0] + 1]

    return [boxes[i].astype(int).tolist() for i in keep]


# ──────────────────────────────────────────────
#  Load
# ──────────────────────────────────────────────
IMAGE_PATH = r"D:\AVIS\Manager\output\DWG\GROUND FLOOR\filtered_floor_plan.png"

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Cannot open image: {IMAGE_PATH}")

H, W = image.shape[:2]


# ──────────────────────────────────────────────
#  STEP 1 — Remove large dark background blobs
# ──────────────────────────────────────────────
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
#  STEP 2 — Non-white mask (no MORPH_OPEN — preserves thin lines)
# ──────────────────────────────────────────────
def build_nonwhite_mask(img, brightness_thresh=210, saturation_thresh=25):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s_ch, v_ch = cv2.split(hsv)
    dark_mask    = (v_ch  < brightness_thresh).astype(np.uint8) * 255
    colored_mask = (s_ch  > saturation_thresh).astype(np.uint8) * 255
    return cv2.bitwise_or(dark_mask, colored_mask)

mask = build_nonwhite_mask(work)


# ──────────────────────────────────────────────
#  STEP 3 — Close gaps (joins box borders without eroding thin lines)
# ──────────────────────────────────────────────
close_kernel  = np.ones((5, 5), np.uint8)
mask_closed   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
dilate_kernel = np.ones((2, 2), np.uint8)
mask_final    = cv2.dilate(mask_closed, dilate_kernel, iterations=1)


# ──────────────────────────────────────────────
#  STEP 4 — Contours (outer only)
# ──────────────────────────────────────────────
contours, hierarchy = cv2.findContours(mask_final, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

MIN_W    = 15
MIN_H    = 15
MIN_AREA = 150
MAX_AREA = 60_000

raw_boxes = []
if hierarchy is not None:
    hier = hierarchy[0]
    for i, cnt in enumerate(contours):
        if hier[i][3] != -1:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if w < MIN_W or h < MIN_H:
            continue
        if area < MIN_AREA or area > MAX_AREA:
            continue
        raw_boxes.append([x, y, w, h])


# ──────────────────────────────────────────────
#  STEP 5 — Suppress overlapping / contained boxes
#
#  Tune these if needed:
#    iou_thresh     — lower = stricter overlap removal  (default 0.3)
#    contain_thresh — lower = removes boxes that are even partially inside another (default 0.6)
# ──────────────────────────────────────────────
final_boxes = suppress_boxes(raw_boxes, iou_thresh=0.3, contain_thresh=0.6)

result = image.copy()
for (x, y, w, h) in final_boxes:
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)

print(f"Detected {len(final_boxes)} shapes after suppression")

cv2.imwrite("debug_raw_mask.png",    mask)
cv2.imwrite("debug_closed_mask.png", mask_final)
cv2.imwrite("detected_shapes.png",   result)

cv2.imshow("Raw Mask",         mask)
cv2.imshow("Closed Mask",      mask_final)
cv2.imshow("Detection Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()