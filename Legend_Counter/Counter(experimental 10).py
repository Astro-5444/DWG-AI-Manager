import base64
import collections
import math
import re
import sys
import time
import requests
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ══════════════════════════════════════════════════════════════════════════════
#  RUN CONFIG  ← edit these two paths
# ══════════════════════════════════════════════════════════════════════════════

SYMBOL_PATH     = r"D:\AVIS\Manager\output\small_test\symbols\symbol_4.png"
FLOOR_PLAN_PATH = r"D:\AVIS\Manager\output\small_test\43001-AJI-04-DWG-IC-BL1-210004-000\floor_plan.png"

USE_AI = True
DEBUG  = True

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CONFIG
# ══════════════════════════════════════════════════════════════════════════════
CFG = {
    # AI server
    "ai_url":         "http://localhost:8080/v1/chat/completions",
    "ai_model":       "qwen-3.5vl-Q4",
    "ai_timeout":     60,
    "ai_max_tokens":  1024,
    "ai_temperature": 0.0,

    # COLOR matching filters
    "color_tolerance":      60,
    "color_top_n":          60,

    # Verification / crops
    "crop_pad":         15,
    "batch_size":       6,
    "crop_size_norm":   128,
    "target_size_norm": 160,

    # Size filtering constraints
    "size_smaller_thresh": 0.8,
    "size_bigger_thresh":  5,

    # Hybrid Verification
    "match_threshold_high": 0.60,
    "match_threshold_low":  0.30,
    "match_scales":         [0.9, 1.0, 1.1],
    "match_angles":         [0, 90, 180, 270],
}

# ──────────────────────────────────────────────────────────────────────────────
_DEBUG_DIR = Path("adaptive_debug")

def _save_debug(name: str, img: np.ndarray) -> None:
    _DEBUG_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(_DEBUG_DIR / name), img)

def _img_to_b64(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("imencode failed")
    return base64.b64encode(buf.tobytes()).decode()

def _handle_alpha(img: np.ndarray, bg_color: tuple = (255, 255, 255)) -> np.ndarray:
    """Composite RGBA symbol onto a neutral background (default white)."""
    if img is None or img.shape[2] < 4:
        return img
    channels = cv2.split(img)
    b, g, r, a = channels
    alpha = a.astype(float) / 255.0
    res = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i, bg_v in enumerate(bg_color):
        res[:, :, i] = (alpha * channels[i] + (1.0 - alpha) * bg_v).astype(np.uint8)
    return res

def call_vlm(messages: list[dict], max_tokens: int | None = None) -> str:
    payload = {
        "model":       CFG["ai_model"],
        "max_tokens":  max_tokens or CFG["ai_max_tokens"],
        "temperature": CFG["ai_temperature"],
        "messages":    messages,
    }
    r = requests.post(CFG["ai_url"], json=payload, timeout=CFG["ai_timeout"])
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ══════════════════════════════════════════════════════════════════════════════
#  THE_TEST_4 EXTRACTION METHODS
# ══════════════════════════════════════════════════════════════════════════════

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

def build_nonwhite_mask(img, brightness_thresh=210, saturation_thresh=25):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s_ch, v_ch = cv2.split(hsv)
    dark_mask    = (v_ch < brightness_thresh).astype(np.uint8) * 255
    colored_mask = (s_ch > saturation_thresh).astype(np.uint8) * 255
    return cv2.bitwise_or(dark_mask, colored_mask)

def get_symbol_size(symbol_path):
    sym = cv2.imread(symbol_path)
    if sym is None:
        raise FileNotFoundError(f"Cannot open symbol: {symbol_path}")
    sym_mask = build_nonwhite_mask(sym)
    close_k  = np.ones((5, 5), np.uint8)
    sym_mask  = cv2.morphologyEx(sym_mask, cv2.MORPH_CLOSE, close_k)
    contours, _ = cv2.findContours(sym_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = sym.shape[:2]
        print(f"[WARN] No contour found in symbol — using full image size: {w}x{h}")
        return w, h
    all_pts     = np.concatenate(contours, axis=0)
    x, y, w, h  = cv2.boundingRect(all_pts)
    print(f"[INFO] Symbol tight bounding box: {w}x{h} px  (at {x},{y})")
    return w, h


# ══════════════════════════════════════════════════════════════════════════════
#  HYBRID VERIFIER
# ══════════════════════════════════════════════════════════════════════════════

def verify_hybrid(
    candidates:  list[dict],
    working_bgr: np.ndarray,
    target_bgr:  np.ndarray,
    debug:       bool = False,
) -> list[dict]:
    """First use Template Matching (CV), then fallback to AI for lower confidence."""
    print(f"\n[4] Hybrid Verification ...")

    # 4a. Run Template Matching on all crops
    confirmed_cv = []
    to_verify_vlm = []

    thr_high = CFG["match_threshold_high"]
    thr_low  = CFG["match_threshold_low"]
    scales   = CFG["match_scales"]
    angles   = CFG["match_angles"]

    print(f"    Step 4a: Template Matching (CV) ...")
    for i, cand in enumerate(candidates):
        # Crop region from the preprocessed floor plan
        crop = _padded_crop(working_bgr, cand["bbox"], 5)
        ch, cw = crop.shape[:2]

        # Multi-scale, Multi-rotation match
        max_score = 0.0
        for scale in scales:
            for angle in angles:
                # Rotate target
                if angle == 90:  target = cv2.rotate(target_bgr, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180: target = cv2.rotate(target_bgr, cv2.ROTATE_180)
                elif angle == 270: target = cv2.rotate(target_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
                else: target = target_bgr

                # Resize target
                th, tw = target.shape[:2]
                nw, nh = int(tw * scale), int(th * scale)
                if nh >= ch or nw >= cw or nh < 5 or nw < 5:
                    continue
                tmpl = cv2.resize(target, (nw, nh), interpolation=cv2.INTER_AREA)

                # Matching
                res = cv2.matchTemplate(crop, tmpl, cv2.TM_CCOEFF_NORMED)
                _, peak, _, _ = cv2.minMaxLoc(res)
                max_score = max(max_score, peak)

        print(f"      Crop {i+1:2d} -> CV Score: {max_score:.2f}", end="")
        cand["score"] = max_score

        if max_score >= thr_high:
            print(" -> [ACCEPTED (CV)]")
            cand["confirmed_by"] = "CV"
            confirmed_cv.append(cand)
        elif max_score >= thr_low:
            print(" -> [TO VERIFY (AI)]")
            to_verify_vlm.append(cand)
        else:
            print(" -> [REJECTED]")

    print(f"    Confirmed (CV)  : {len(confirmed_cv)}")
    print(f"    To Verify (AI)  : {len(to_verify_vlm)}")

    # 4b. Run VLM on candidates in the 0.3 - 0.6 range
    confirmed_vlm = []
    if to_verify_vlm and USE_AI:
        print(f"    Step 4b: VLM Fallback ...")
        # Reuse existing batch function
        vlm_results = verify_with_vlm(to_verify_vlm, working_bgr, target_bgr, debug)
        for cand in vlm_results:
            cand["confirmed_by"] = "AI"
            confirmed_vlm.append(cand)

    # Combine
    return confirmed_cv + confirmed_vlm

_VERIFY_SYSTEM = "You are an automated visual inspection system. Match TARGET symbols to CROP regions."

_VERIFY_PROMPT = """\
# INPUT
- TARGET: Reference symbol in RED box (top)
- CROPS:  Numbered candidates in BLUE boxes (bottom grid)

# RULES
1. IGNORE ORIENTATION — rotation never matters.
2. Find the CORE SYMBOL in each image (primary geometric shape + fill).
3. IGNORE attachments: arrows, leader lines, stems, dimension lines.
4. IGNORE overlays: CAD background lines, hatching, text.
5. Match ONLY: core shape geometry + internal fill pattern.

# OUTPUT FORMAT  (mandatory)
<thinking>
Target Core Symbol: [shape] with [fill pattern]
Crop N: [shape] with [fill] -> Match: Yes/No
...
</thinking>
<Answer>
[Comma-separated crop numbers that match. Empty if none.]
</Answer>
"""

def _padded_crop(img: np.ndarray, bbox: tuple, pad: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    x, y, w, h = bbox
    return img[max(0,y-pad):min(ih,y+h+pad), max(0,x-pad):min(iw,x+w+pad)]

def _build_batch_image(target_bgr: np.ndarray, crops: list, indices: list) -> np.ndarray:
    ts = CFG["target_size_norm"]
    timg = cv2.resize(target_bgr, (ts, ts), interpolation=cv2.INTER_AREA)
    timg = cv2.copyMakeBorder(timg, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(0, 0, 255))
    th, tw = timg.shape[:2]

    n    = len(crops)
    cols = min(n, 3)
    rows = math.ceil(n / cols)
    cs   = CFG["crop_size_norm"]
    cw   = cs + 4;  ch = cs + 4
    pad  = 10;      txt_h = 25

    canvas_w = max(tw, cols*(cw+pad)+pad)
    canvas_h = th + 50 + rows*(ch+txt_h+pad) + pad
    canvas   = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    cv2.putText(canvas, "TARGET SYMBOL", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    tx = (canvas_w - tw) // 2
    canvas[30:30+th, tx:tx+tw] = timg

    gy = 30 + th + 20
    cv2.putText(canvas, "CANDIDATE CROPS", (10, gy-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    for i, (crop, idx) in enumerate(zip(crops, indices)):
        row, col = i // cols, i % cols
        rc  = cv2.resize(crop, (cs, cs), interpolation=cv2.INTER_AREA)
        rc  = cv2.copyMakeBorder(rc, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255,0,0))
        x0  = pad + col*(cw+pad)
        y0  = gy  + row*(ch+txt_h+pad)
        cv2.putText(canvas, f"Crop {idx}", (x0, y0+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        canvas[y0+txt_h:y0+txt_h+ch, x0:x0+cw] = rc

    return canvas

def _ask_vlm_batch(batch_img: np.ndarray) -> list[int]:
    messages = [
        {"role": "system", "content": _VERIFY_SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{_img_to_b64(batch_img)}"}},
            {"type": "text", "text": _VERIFY_PROMPT},
        ]},
    ]
    try:
        content = call_vlm(messages)
        m = re.search(r"<Answer>(.*?)</Answer>", content, re.IGNORECASE | re.DOTALL)
        if m:
            ans = m.group(1).strip()
            if not ans:
                return []
            return [int(re.sub(r"\D", "", t))
                    for t in ans.split(",") if re.sub(r"\D", "", t)]
        print("[WARN] No <Answer> block in VLM response")
        return []
    except Exception as exc:
        print(f"[WARN] VLM verify failed: {exc}")
        return []

def verify_with_vlm(
    candidates:  list[dict],
    working_bgr: np.ndarray,
    target_bgr:  np.ndarray,
    debug:       bool = False,
) -> list[dict]:
    bs        = CFG["batch_size"]
    confirmed = []

    for i in range(0, len(candidates), bs):
        chunk   = candidates[i:i+bs]
        crops   = [_padded_crop(working_bgr, c["bbox"], CFG["crop_pad"]) for c in chunk]
        indices = list(range(i+1, i+len(chunk)+1))

        print(f"    Batch {i//bs+1}  crops {indices[0]}–{indices[-1]} ...")
        batch_img = _build_batch_image(target_bgr, crops, indices)

        if debug:
            _save_debug(f"04_batch_{i//bs+1}.png", batch_img)

        t0      = time.perf_counter()
        matches = _ask_vlm_batch(batch_img)
        print(f"      → matches={matches}  ({time.perf_counter()-t0:.1f}s)")

        for m in matches:
            if m in indices:
                confirmed.append(chunk[indices.index(m)])

    return confirmed


# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def save_annotated(
    original_bgr: np.ndarray,
    candidates:   list[dict],
    confirmed:    list[dict],
    out_path:     str,
) -> None:
    vis = original_bgr.copy()
    for c in candidates:
        x, y, w, h = c["bbox"]
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 200, 200), 1)
    for c in confirmed:
        x, y, w, h = c["bbox"]
        color = (0, 255, 0) if c.get("confirmed_by") == "CV" else (255, 100, 0)
        cv2.rectangle(vis, (x, y), (x+w, y+h), color, 3)
        lbl = f"MATCH ({c.get('confirmed_by', '??')})"
        cv2.putText(vis, lbl, (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(vis, f"Candidates : {len(candidates)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
    cv2.putText(vis, f"Confirmed  : {len(confirmed)}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imwrite(out_path, vis)
    print(f"\n[OUTPUT] → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run() -> None:
    print(f"\n{'='*60}")
    print(f" The_TEST_4 + Hybrid Verification Pipeline")
    print(f"{'='*60}\n")
    
    # Target
    target_raw = cv2.imread(SYMBOL_PATH, cv2.IMREAD_UNCHANGED)
    if target_raw is None:
        sys.exit(f"[ERROR] Cannot open symbol: {SYMBOL_PATH}")
    target_bgr = _handle_alpha(target_raw)

    # Origin
    original_fp = cv2.imread(FLOOR_PLAN_PATH)
    if original_fp is None:
        sys.exit(f"[ERROR] Cannot open floor plan: {FLOOR_PLAN_PATH}")

    # Step 1: Filter symbol colors
    print("\n[1] Filtering floor plan with symbol colors ...")
    symbol = Image.open(SYMBOL_PATH).convert("RGB")
    
    # Avoid getdata deprecation by using np array
    sym_arr = np.array(symbol)
    pixels  = sym_arr.reshape(-1, 3).tolist()
    pixels  = [tuple(p) for p in pixels]

    color_counts     = collections.Counter(pixels)
    background_color = color_counts.most_common(1)[0][0]
    sorted_colors    = color_counts.most_common()
    sorted_colors    = [c for c in sorted_colors if c[0] != background_color]
    top_colors       = sorted_colors[:CFG["color_top_n"]]
    symbol_colors    = [c[0] for c in top_colors]

    print(f"    Background color    : {background_color}")
    print(f"    Symbol colors count : {len(symbol_colors)}")

    floor  = Image.open(FLOOR_PLAN_PATH).convert("RGB")
    fp     = np.array(floor, dtype=np.int32)
    result = np.full_like(fp, [255, 255, 255], dtype=np.int32)
    
    for sym_color in symbol_colors:
        diff  = np.abs(fp - sym_color)
        match = np.all(diff <= CFG["color_tolerance"], axis=2)
        result[match] = fp[match]

    # Convert to BGR for cv2 operations
    filtered_bgr = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR)
    if DEBUG:
        _save_debug("01_filtered_floor_plan.png", filtered_bgr)

    # Step 2: Region Extraction 
    print("\n[2] Extracting Regions ...")
    sym_w, sym_h = get_symbol_size(SYMBOL_PATH)

    min_w = sym_w * (1.0 - CFG["size_smaller_thresh"])
    max_w = sym_w * (1.0 + CFG["size_bigger_thresh"])
    min_h = sym_h * (1.0 - CFG["size_smaller_thresh"])
    max_h = sym_h * (1.0 + CFG["size_bigger_thresh"])

    print(f"    Allowed W: {min_w:.0f} – {max_w:.0f} px")
    print(f"    Allowed H: {min_h:.0f} – {max_h:.0f} px")

    H, W = filtered_bgr.shape[:2]

    # Remove dark background
    gray = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2GRAY)
    _, dark_bg_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    bg_contours, _  = cv2.findContours(dark_bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bg_mask = np.zeros((H, W), dtype=np.uint8)
    for cnt in bg_contours:
        if cv2.contourArea(cnt) > 10_000:
            cv2.drawContours(bg_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    work = filtered_bgr.copy()
    work[bg_mask == 255] = 255

    # Non-white mask
    mask = build_nonwhite_mask(work)

    # Close gaps + gentle dilate
    close_kernel  = np.ones((5, 5), np.uint8)
    mask_closed   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    dilate_kernel = np.ones((2, 2), np.uint8)
    mask_final    = cv2.dilate(mask_closed, dilate_kernel, iterations=1)

    # Contours
    contours, hierarchy = cv2.findContours(mask_final, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes     = []
    rejected_size = 0

    if hierarchy is not None:
        hier = hierarchy[0]
        for i, cnt in enumerate(contours):
            if hier[i][3] != -1:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if w < 15 or h < 15 or area < 150:
                continue
            if w < min_w or w > max_w or h < min_h or h > max_h:
                rejected_size += 1
                continue
            raw_boxes.append([x, y, w, h])

    final_boxes = suppress_boxes(raw_boxes, iou_thresh=0.3, contain_thresh=0.6)
    print(f"    Rejected by size filter   : {rejected_size}")
    print(f"    Detected after suppression: {len(final_boxes)}")

    if DEBUG:
        _save_debug("02_raw_mask.png", mask)
        _save_debug("02_mask_final.png", mask_final)

    # FORMAT CANDIDATES FOR HYBRID VERIFIER
    candidates = []
    # Setting an arbitrary score = 1.0 for all
    for (x, y, w, h) in final_boxes:
        candidates.append({"bbox": (x, y, w, h), "score": 1.0})

    if not candidates:
        print("[INFO] No candidates found. Done.")
        return

    # 3. Hybrid Verification
    confirmed = verify_hybrid(candidates, filtered_bgr, target_bgr, debug=DEBUG)

    # 4. Save result
    out = Path(FLOOR_PLAN_PATH).stem + "_result.png"
    save_annotated(original_fp, candidates, confirmed, out)

    print(f"\n{'─'*60}")
    print(f"  Candidates : {len(candidates)}")
    print(f"  Confirmed  : {len(confirmed)}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    run()