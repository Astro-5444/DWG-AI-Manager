import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests, base64, time, math, multiprocessing, os, re
from io import BytesIO
from typing import List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from skimage.metrics import structural_similarity as ssim

Image.MAX_IMAGE_PIXELS = 500000000

# ==============================
# CONFIGURATION
# ==============================
API_KEY    = "ak_OFIKwiNCW2UcDWkLrVRhMR-tVb9SIwaGvGeGueDk1tM"
BASE_URL   = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "qwen-3.5vl-Q4"

SYMBOL_PATH     = r"D:\AVIS\Manager\output\small_test\symbols\symbol_4.png"
FLOOR_PLAN_PATH = r"D:\AVIS\Manager\output\small_test\43001-AJI-04-DWG-IC-BL1-210004-000\floor_plan.png"
OUTPUT_DIR      = "debug_output"
CROP_PADDING    = 3

# ── Candidate warning threshold ───────────────────────────────────────────────
# If candidates exceed this after template matching, warn user and stop.
CANDIDATE_WARNING_LIMIT = 300

# ── LLM Mode ─────────────────────────────────────────────────────────────────
# "batch"      → grid of BATCH_SIZE crops in one call, AI replies "1,3,5"
# "individual" → one crop per call, AI replies "YES" or "NO"
# "both"       → runs BOTH and compares speed + result (for benchmarking)
LLM_MODE       = "batch"   # "batch" | "individual" | "both"
LLM_BATCH_SIZE = 6

# ── Confidence thresholds ─────────────────────────────────────────────────────
HIGH_CONF_AUTO   = 0.93
LOW_CONF_AUTO    = 0.60
CV_CONFIRM_SCORE = 0.55
CV_REJECT_SCORE  = 0.30
# ─────────────────────────────────────────────────────────────────────────────

# ── Color Filter Optimization ────────────────────────────────────────────────
USE_COLOR_TILE_FILTER = True

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
# Symbol analysis helpers
# ==============================

def analyze_symbol(symbol_bgr: np.ndarray) -> dict:
    """
    Analyze the symbol to detect if CV verification will be unreliable.
    Returns a dict with warnings and detected properties.
    """
    gray   = cv2.cvtColor(symbol_bgr, cv2.COLOR_BGR2GRAY)
    hsv    = cv2.cvtColor(symbol_bgr, cv2.COLOR_BGR2HSV)
    h, w   = gray.shape[:2]

    warnings = []

    # ── 1. Check if symbol is a simple rectangle (no texture) ────────────────
    edges      = cv2.Canny(gray, 50, 150)
    contours,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    is_simple_rect = False
    if contours:
        largest = max(contours, key=cv2.contourArea)
        peri    = cv2.arcLength(largest, True)
        approx  = cv2.approxPolyDP(largest, 0.04 * peri, True)
        if len(approx) == 4:
            is_simple_rect = True
            warnings.append("⚠️  Symbol is a simple rectangle — SSIM & ORB will be unreliable")

    # ── 2. Check if symbol has distinctive color ──────────────────────────────
    # Detect dominant hue
    mask          = cv2.inRange(hsv, (0,30,30), (180,255,255))  # non-grey pixels
    colored_ratio = np.count_nonzero(mask) / (h * w)
    dominant_hue  = None
    is_yellow     = False
    if colored_ratio > 0.05:
        hues = hsv[:,:,0][mask > 0]
        dominant_hue = int(np.median(hues))
        # Yellow in HSV: hue 20–35
        if 20 <= dominant_hue <= 35:
            is_yellow = True

    # ── 3. ORB keypoint count ─────────────────────────────────────────────────
    orb     = cv2.ORB_create(nfeatures=500)
    kp, _   = orb.detectAndCompute(gray, None)
    n_kp    = len(kp) if kp else 0
    if n_kp < 10:
        warnings.append(f"⚠️  Only {n_kp} ORB keypoints detected — ORB verification will score ~0.00")

    return {
        'is_simple_rect': is_simple_rect,
        'is_yellow':      is_yellow,
        'dominant_hue':   dominant_hue,
        'colored_ratio':  colored_ratio,
        'orb_keypoints':  n_kp,
        'warnings':       warnings,
        'cv_reliable':    not is_simple_rect and n_kp >= 10,
    }


# ==============================
# Color filtering helpers
# ==============================

def _get_red_mask(image):
    """Returns a binary mask of red pixels in BGR image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Red spans 0-10 and 160-180 in OpenCV HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    return cv2.bitwise_or(mask1, mask2)

def _is_red_symbol(symbol_bgr):
    """Checks if the symbol contains a significant amount of red."""
    mask = _get_red_mask(symbol_bgr)
    total = symbol_bgr.shape[0] * symbol_bgr.shape[1]
    return (cv2.countNonZero(mask) / total) > 0.05

def extract_top_hsv_colors(bgr_img: np.ndarray, num_colors=2, min_saturation=30, min_value=30):
    import collections
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    flat = hsv.reshape(-1, 3)
    
    # Filter out white/black/grey by requiring minimum saturation and value
    mask = (flat[:, 1] > min_saturation) & (flat[:, 2] > min_value)
    colored = flat[mask]
    
    if len(colored) == 0:
        return []
        
    hues = colored[:, 0]
    hue_counts = collections.Counter(hues)
    return [int(hue) for hue, count in hue_counts.most_common(num_colors)]

def highlight_hsv_colors(image: np.ndarray, target_hues: list,
                         color_range_pct=0.30, boost_to=(0, 0, 255),
                         darken_others=True, darken_factor=0.9):
    """
    Highlights pixels matching target hues within a specified percentage range.
    color_range_pct: 0.10 means 10% of the hue range (which is 18 in OpenCV's 0-180 scale).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    combined_mask = np.zeros((h, w), dtype=bool)

    # Calculate tolerance in Hue scale (0-180)
    tolerance = int(180 * color_range_pct)

    for thue in target_hues:
        # Calculate angular distance for Hue (circular 0-180)
        hue_channel = hsv[:, :, 0].astype(np.int32)
        diff = np.abs(hue_channel - thue)
        diff = np.minimum(diff, 180 - diff)
        
        # Only match if sat/val are also high enough to actually have color
        mask = (diff <= tolerance) & (hsv[:, :, 1] > 30) & (hsv[:, :, 2] > 30)
        combined_mask = combined_mask | mask
        
    result = image.copy()
    result[combined_mask] = boost_to
    
    if darken_others:
        not_mask = ~combined_mask
        result[not_mask] = (image[not_mask].astype(np.float32) * darken_factor).astype(np.uint8)
        
    return result

def extract_icon_colors(icon_img, brightness_threshold=240):
    if icon_img is None:
        return []
    pixels = icon_img.reshape(-1, 3)
    mask   = np.any(pixels < brightness_threshold, axis=1)
    fg     = pixels[mask]
    if len(fg) == 0:
        return []
    quantized     = (fg.astype(np.int32) // 16) * 16
    unique_colors = np.unique(quantized, axis=0).astype(np.uint8)
    return unique_colors


def filter_by_color(image, icon_colors, tolerance=30):
    if len(icon_colors) == 0:
        print("No colors detected in icon — skipping color filter.")
        return image.copy()
    h, w          = image.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for color in icon_colors:
        lower = np.clip(color.astype(np.int32) - tolerance,      0, 255).astype(np.uint8)
        upper = np.clip(color.astype(np.int32) + 15 + tolerance, 0, 255).astype(np.uint8)
        mask  = cv2.inRange(image, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    result = np.full_like(image, 255)
    result[combined_mask > 0] = image[combined_mask > 0]
    return result


def filter_by_hsv_color(image: np.ndarray, symbol_bgr: np.ndarray,
                         tolerance_h=15, tolerance_sv=40) -> np.ndarray:
    """
    Tighter color filter using HSV — much better for solid-color symbols like yellow.
    Keeps only pixels whose hue/saturation/value is close to the symbol's dominant color.
    """
    hsv_sym  = cv2.cvtColor(symbol_bgr, cv2.COLOR_BGR2HSV)
    hsv_img  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Get median hue/sat/val of non-white symbol pixels
    h_s, w_s = hsv_sym.shape[:2]
    flat      = hsv_sym.reshape(-1, 3)
    sat_mask  = flat[:, 1] > 30   # ignore near-grey pixels
    if not np.any(sat_mask):
        print("  HSV filter: symbol appears grey/white — falling back to RGB filter")
        return filter_by_color(image, extract_icon_colors(symbol_bgr), tolerance=30)

    colored   = flat[sat_mask]
    med_h     = int(np.median(colored[:, 0]))
    med_s     = int(np.median(colored[:, 1]))
    med_v     = int(np.median(colored[:, 2]))

    print(f"  HSV filter: dominant HSV = ({med_h}, {med_s}, {med_v})")

    lo_h = max(0,   med_h - tolerance_h)
    hi_h = min(180, med_h + tolerance_h)
    lo_s = max(0,   med_s - tolerance_sv)
    hi_s = min(255, med_s + tolerance_sv)
    lo_v = max(0,   med_v - tolerance_sv)
    hi_v = min(255, med_v + tolerance_sv)

    lower = np.array([lo_h, lo_s, lo_v], dtype=np.uint8)
    upper = np.array([hi_h, hi_s, hi_v], dtype=np.uint8)
    mask  = cv2.inRange(hsv_img, lower, upper)

    result = np.full_like(image, 255)
    result[mask > 0] = image[mask > 0]

    kept_pct = 100 * np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1])
    print(f"  HSV filter: {kept_pct:.1f}% of floor plan pixels kept")
    return result


# ==============================
# Template matching workers
# ==============================

def _match_angle(task):
    # Unpack task (handles optional valid_tiles for compatibility)
    if len(task) >= 5:
        angle, icon, image, threshold, valid_tiles = task[:5]
    else:
        angle, icon, image, threshold = task
        valid_tiles = None

    # For _match_angle, if the whole image has no red, skip the whole task
    if USE_COLOR_TILE_FILTER and valid_tiles is not None and not valid_tiles:
        # In this context, if we passed an empty set/None purposefully, 
        # it means no red was found in the whole image.
        return []

    ih, iw  = icon.shape[:2]
    matches = []
    if ih > image.shape[0] or iw > image.shape[1]:
        return matches
    result    = cv2.matchTemplate(image, icon, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    for pt in zip(*locations[::-1]):
        matches.append({'x': int(pt[0]), 'y': int(pt[1]),
                        'w': iw, 'h': ih, 'angle': angle,
                        'confidence': float(result[pt[1], pt[0]])})
    print(f"  Angle {angle:>6}: {len(matches)} raw hits")
    return matches


def _match_angle_tiled(task, tile_size=2000, overlap=200):
    # Unpack task
    if len(task) >= 5:
        angle, icon, image, threshold, valid_tile_indices = task[:5]
    else:
        angle, icon, image, threshold = task
        valid_tile_indices = None

    ih, iw  = icon.shape[:2]
    h, w    = image.shape[:2]
    matches = []
    stride  = tile_size - overlap
    
    skipped_count = 0
    total_tiles = 0
    
    for ty in range(math.ceil((h - overlap) / stride)):
        for tx in range(math.ceil((w - overlap) / stride)):
            total_tiles += 1
            # Skip if color filter is enabled and this tile index is not in valid list
            if USE_COLOR_TILE_FILTER and valid_tile_indices is not None:
                if (ty, tx) not in valid_tile_indices:
                    skipped_count += 1
                    continue

            ys, xs = ty * stride, tx * stride
            ye, xe = min(ys + tile_size, h), min(xs + tile_size, w)
            tile   = image[ys:ye, xs:xe]
            if ih > tile.shape[0] or iw > tile.shape[1]:
                continue
            result    = cv2.matchTemplate(tile, icon, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)
            for pt in zip(*locations[::-1]):
                matches.append({'x': int(pt[0]) + xs, 'y': int(pt[1]) + ys,
                                'w': iw, 'h': ih, 'angle': angle,
                                'confidence': float(result[pt[1], pt[0]])})
    
    if skipped_count > 0:
        print(f"  Angle {angle:>6}: {len(matches)} hits (Skipped {skipped_count}/{total_tiles} non-red tiles)")
    else:
        print(f"  Angle {angle:>6}: {len(matches)} raw hits (tiled)")
    return matches


# ==============================
# CV Verifier
# ==============================

class CVVerifier:
    ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]

    def __init__(self, symbol_bgr):
        self.symbol_gray = cv2.cvtColor(symbol_bgr, cv2.COLOR_BGR2GRAY)
        self.symbol_bgr  = symbol_bgr
        
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Pre-generate ALL templates (Normal + H/V/B Flips) for all angles
        self._templates = {}
        # 1. Normal
        for a in self.ANGLES:
            self._templates[a] = self._rotate(self.symbol_gray, a)
        
        # 2. Flips: Horizontal (1), Vertical (0), Both (-1)
        # Note: We use the same suffixes as IconCounter for easy lookup
        for code, suffix in [(1, 'fH'), (0, 'fV'), (-1, 'fB')]:
            flipped = cv2.flip(self.symbol_gray, code)
            for a in self.ANGLES:
                self._templates[f"{a}_{suffix}"] = self._rotate(flipped, a)

        # Pre-compute ORB for both Normal and Mirrored (H-flip)
        # (V-flip and Both are covered by H-flip descriptors since ORB is rotation invariant)
        self.sym_kp, self.sym_des = self.orb.detectAndCompute(self.symbol_gray, None)
        
        mirrored_gray = cv2.flip(self.symbol_gray, 1)
        self.sym_kp_m, self.sym_des_m = self.orb.detectAndCompute(mirrored_gray, None)
        
        self.sym_hist = self._hist(symbol_bgr)

    @staticmethod
    def _rotate(img, angle):
        h, w = img.shape[:2]
        M    = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        cos, sin = abs(M[0,0]), abs(M[0,1])
        nw = int(h*sin + w*cos); nh = int(h*cos + w*sin)
        M[0,2] += nw/2 - w//2;  M[1,2] += nh/2 - h//2
        return cv2.warpAffine(img, M, (nw,nh), borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    @staticmethod
    def _hist(bgr):
        h = cv2.calcHist([bgr],[0,1,2],None,[8,8,8],[0,256,0,256,0,256])
        cv2.normalize(h,h); return h.flatten()

    def _ssim_score(self, crop_gray, angle_key=None):
        best = 0.0
        
        # If we have a specific angle_key from template matching, check that first (Optimization)
        # Use a small window around it if needed, but primarily trust the key.
        search_keys = [angle_key] if angle_key in self._templates else self._templates.keys()
        
        for k in search_keys:
            sym_rot = self._templates[k]
            ch, cw  = crop_gray.shape[:2]
            sh, sw  = sym_rot.shape[:2]
            if sh==0 or sw==0: continue
            
            # Since crops are tightly mapped to template sizes in matching, scale should be near 1.0
            scale = min(cw/sw, ch/sh)
            if scale<=0: continue
            
            resized = cv2.resize(sym_rot, (max(1,int(sw*scale)), max(1,int(sh*scale))),
                                 interpolation=cv2.INTER_AREA)
            rh, rw  = resized.shape[:2]
            if rh>ch or rw>cw or min(rh,rw)<7: continue
            
            y_off = (ch-rh)//2; x_off = (cw-rw)//2
            region = crop_gray[y_off:y_off+rh, x_off:x_off+rw]
            if region.shape != resized.shape: continue
            try:
                s,_ = ssim(region, resized, full=True)
                best = max(best, s)
            except: pass
        return max(0.0, best)

    def _orb_score(self, crop_gray):
        if self.sym_des is None: return 0.0
        kp2, des2 = self.orb.detectAndCompute(crop_gray, None)
        if des2 is None or len(kp2)==0: return 0.0
        
        best_ratio = 0.0
        # Check against both Normal and Mirrored descriptors
        for (des_ref, kp_ref) in [(self.sym_des, self.sym_kp), (self.sym_des_m, self.sym_kp_m)]:
            if des_ref is None: continue
            try:
                matches = self.bf.match(des_ref, des2)
                good    = [m for m in matches if m.distance < 50]
                ratio   = len(good) / max(len(kp_ref), 1)
                best_ratio = max(best_ratio, ratio)
            except: pass
        return min(1.0, best_ratio)

    def _hist_score(self, crop_bgr):
        ch = self._hist(crop_bgr)
        return float(cv2.compareHist(
            self.sym_hist.reshape(-1,1).astype(np.float32),
            ch.reshape(-1,1).astype(np.float32),
            cv2.HISTCMP_CORREL))

    def score(self, crop_bgr, angle_key=None):
        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        ssim_s    = self._ssim_score(crop_gray, angle_key=angle_key)
        orb_s     = self._orb_score(crop_gray)
        hist_s    = max(0.0, self._hist_score(crop_bgr))
        combined  = ssim_s*0.50 + orb_s*0.30 + hist_s*0.20
        return {'ssim':ssim_s,'orb':orb_s,'hist':hist_s,'combined':combined}


# ==============================
# Image helpers
# ==============================

def image_to_base64(img: Image.Image) -> str:
    buf = BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_grid(crops, cols=3, cell_size=256, gap=4):
    rows   = math.ceil(len(crops) / cols)
    grid_w = cols*cell_size + (cols-1)*gap
    grid_h = rows*cell_size + (rows-1)*gap
    grid   = Image.new("RGB", (grid_w, grid_h), (220,220,220))
    draw   = ImageDraw.Draw(grid)
    try:    font = ImageFont.truetype("arial.ttf", 22)
    except: font = None
    for i, crop in enumerate(crops):
        col   = i%cols; row = i//cols
        xo    = col*(cell_size+gap); yo = row*(cell_size+gap)
        thumb = crop.copy()
        thumb.thumbnail((cell_size-4, cell_size-4), Image.LANCZOS)
        tw,th = thumb.size
        grid.paste(thumb, (xo+(cell_size-tw)//2, yo+(cell_size-th)//2))
        draw.rectangle((xo,yo,xo+cell_size-1,yo+cell_size-1), outline=(80,80,200), width=2)
        draw.rectangle((xo,yo,xo+28,yo+26), fill=(80,80,200))
        draw.text((xo+4,yo+2), str(i+1), fill="white", font=font)
    return grid


def create_combined_image(symbol_pil, crop_pils, batch_idx=0, label=""):
    """Creates a single PIL image with the Target on the left and a Grid of crops on the right."""
    CELL=256; GAP=4; PAD=12
    sw,sh  = symbol_pil.size
    # Upscale symbol for better visibility
    sym_up = symbol_pil.resize((sw*3,sh*3), Image.BICUBIC)
    sym_up.thumbnail((CELL*2,CELL*2), Image.LANCZOS)
    
    # Create the grid of crops
    grid = _make_grid(crop_pils, cols=min(3,len(crop_pils)), cell_size=CELL, gap=GAP)
    
    total_w = sym_up.width + PAD*3 + grid.width
    total_h = max(sym_up.height, grid.height) + PAD*2 + 40
    canvas  = Image.new("RGB", (total_w, total_h), (245,245,245))
    draw    = ImageDraw.Draw(canvas)
    
    try:    tf = ImageFont.truetype("arial.ttf", 18)
    except: tf = None
    
    title = f"Batch {batch_idx+1}  {label}  |  TARGET (left)  CROPS (right)"
    draw.text((PAD, 6), title, fill=(30,30,30), font=tf)
    
    # Draw RED box around TARGET
    draw.rectangle((PAD-2, 40-2, PAD+sym_up.width+2, 40+sym_up.height+2),
                   outline=(220,80,80), width=4)
    canvas.paste(sym_up, (PAD, 40))
    
    # Grid is already boxed in _make_grid, just paste it
    canvas.paste(grid, (PAD*2+sym_up.width, 40))
    
    return canvas

def save_grid_with_symbol(symbol_pil, crop_pils, batch_idx, out_dir, label=""):
    canvas = create_combined_image(symbol_pil, crop_pils, batch_idx, label)
    path = os.path.join(out_dir, f"batch_{batch_idx+1:03d}_grid{label}.png")
    canvas.save(path)
    return path


# ==============================
# LLM — BATCH mode (grid → numbers)
# ==============================

def llm_batch(symbol_pil, crop_pils, batch_idx=0, retries=2):
    """Send grid image, AI replies with thinking + <Answer> tag. Logs raw response to .txt."""
    n        = len(crop_pils)
    n = len(crop_pils)
    # Build dynamic template lines for thinking block
    crops_template = "\n".join([f"Crop {i+1} Core Symbol: [Shape] with [Fill Pattern] -> Match: [Yes/No] -> [If no, which rule # was broken?]" for i in range(n)])

    # Create single combined image (Target + Grid)
    combined_img = create_combined_image(symbol_pil, crop_pils, batch_idx, label="_batch")

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": f"""
# ROLE
You are an automated visual inspection system. Match TARGET symbols to CROP regions.

# INPUT
- TARGET: Reference symbol in RED box
- CROPS: Numbered candidates in BLUE boxes

# CRITICAL RULES
1. IGNORE ORIENTATION: Rotation never matters.
2. IDENTIFY THE CORE SYMBOL: Find the primary geometric shape (triangle, circle, square, etc.).
3. IGNORE ATTACHMENTS: Disregard arrows, dimension lines, leader lines, stems, connectors, or pointers attached to the symbol.
4. IGNORE OVERLAYS: Disregard colored hatched bars, background CAD lines, and text.
5. MATCH ONLY: Core shape geometry + internal fill pattern.
6. BE SMART: Focus only on the target symbol colors any other colors don`t think about it 

# PROCESS
1. Identify the CORE SYMBOL in TARGET (ignore any attachments or any overlay).
2. Describe its fill pattern (solid, split, hatched, etc.).
3. For each CROP, identify ONLY the core symbol (ignore attachments/overlays).
4. Match based on: Shape + Fill Pattern (orientation irrelevant).

# MUST USE THIS TEMPLATE
<thinking>
*Rules*
list of rules you have.
Target Core Symbol: [Shape] with [Fill Pattern]
Crop 1 Core Symbol: [Shape] with [Fill Pattern] -> Match: [Yes/No] ->[If no what is the number of the rule didn't it break]
Crop 2 Core Symbol: [Shape] with [Fill Pattern] -> Match: [Yes/No] ->[If no what is the number of the rule didn't it break]
Crop 3 Core Symbol: [Shape] with [Fill Pattern] -> Match: [Yes/No] ->[If no what is the number of the rule didn't it break]
</thinking>
<Answer>
[Comma-separated numbers of exact matches only. If none match, leave empty.]
</Answer>

"""},
            {"role": "user", "content": [
                {"type":"image_url", "image_url":{"url":f"data:image/png;base64,{image_to_base64(combined_img)}"}},
            ]},
        ],
        "temperature": 0.0, "max_tokens": 800,
    }
    headers = {"Authorization":f"Bearer {API_KEY}", "Content-Type":"application/json"}

    for attempt in range(retries+1):
        t0_call = time.time()
        try:
            r   = requests.post(BASE_URL, headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            elapsed_call = time.time() - t0_call
            raw = r.json()["choices"][0]["message"]["content"].strip()
            
            # SAVE FULL RESPONSE TO TXT
            log_path = os.path.join(OUTPUT_DIR, f"batch_{batch_idx+1:03d}_ai_response.txt")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"=== BATCH {batch_idx+1} ===\n")
                f.write(f"Call Duration: {elapsed_call:.2f}s\n")
                f.write(f"Candidates: {n}\n")
                f.write("-" * 40 + "\n")
                f.write("RAW RESPONSE:\n")
                f.write(raw)
                f.write("\n" + "-" * 40 + "\n")

            print(f"\n  ┌─ BATCH AI RESPONSE (batch {batch_idx+1}) {'─'*30}")
            print(f"  │  Log saved to: {os.path.basename(log_path)}")
            
            # Extract content from <Answer> tag
            ans_match = re.search(r'<Answer>(.*?)</Answer>', raw, re.DOTALL | re.IGNORECASE)
            answer_text = ans_match.group(1).strip() if ans_match else raw
            
            confirmed = set()
            for tok in answer_text.replace(","," ").split():
                d = "".join(c for c in tok if c.isdigit())
                if d:
                    idx = int(d)
                    if 1 <= idx <= n:
                        confirmed.add(idx)
            
            print(f"  │  Parsed matches: {sorted(confirmed) if confirmed else 'none'}")
            print(f"  └{'─'*44}")
            return confirmed
        except Exception as e:
            # Handle logging failure too
            try:
                log_path = os.path.join(OUTPUT_DIR, f"batch_{batch_idx+1:03d}_error.txt")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"Error at {time.strftime('%H:%M:%S')}: {str(e)}\n")
            except: pass
            print(f"  (batch LLM error attempt {attempt+1}: {e})")
            if attempt < retries: time.sleep(2)
    return None




# ==============================
# LLM — INDIVIDUAL mode (single crop → YES/NO)
# ==============================

    # Create single combined image (Target + Single Crop)
    combined_img = create_combined_image(symbol_pil, [crop_pil], batch_idx=0, label="_ind")

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": """
# ROLE
You are an automated visual inspection system. Match TARGET symbols to CROP regions.

# INPUT
- TARGET: Reference symbol in RED box
- CROP: Candidate region in BLUE box

# CRITICAL RULES
1. IGNORE ORIENTATION: Rotation never matters.
2. IDENTIFY THE CORE SYMBOL: Find the primary geometric shape (triangle, circle, square, etc.).
3. IGNORE ATTACHMENTS: Disregard arrows, dimension lines, leader lines, stems, connectors, or pointers attached to the symbol.
4. IGNORE OVERLAYS: Disregard colored hatched bars, background CAD lines, and text.
5. MATCH ONLY: Core shape geometry + internal fill pattern.
6. BE SMART: Focus only on the target symbol colors; ignore any other colors.

# PROCESS
1. Identify the CORE SYMBOL in TARGET (ignore any attachments or overlays).
2. Describe its fill pattern (solid, split, hatched, etc.).
3. For the CROP, identify ONLY the core symbol (ignore attachments/overlays).
4. Match based on: Shape + Fill Pattern (orientation irrelevant).

# MUST USE THIS TEMPLATE
<thinking>
*Rules*
1. Ignore orientation. 2. Identify core symbol. 3. Ignore attachments. 4. Ignore overlays. 5. Match shape+fill. 6. Focus color.

Target Core Symbol: [Shape] with [Fill Pattern]
Crop 1 Core Symbol: [Shape] with [Fill Pattern] -> Match: [Yes/No] -> [If no, which rule # was broken?]
</thinking>
<Answer>
[1 if it matches, otherwise leave empty.]
</Answer>
"""},
            {"role": "user", "content": [
                {"type":"image_url", "image_url":{"url":f"data:image/png;base64,{image_to_base64(combined_img)}"}},
            ]},
        ],
        "temperature": 0.0, "max_tokens": 250,
    }
    headers = {"Authorization":f"Bearer {API_KEY}", "Content-Type":"application/json"}

    for attempt in range(retries+1):
        t0_call = time.time()
        try:
            r   = requests.post(BASE_URL, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            elapsed_call = time.time() - t0_call
            raw = r.json()["choices"][0]["message"]["content"].strip()
            
            # SAVE RAW RESPONSE
            # Use timestamp or serial if multiple crops? Individual mode usually has many calls.
            # We'll use a unique identifier or just skip detailed logging for individual unless requested, 
            # but user asked for "each batch" - however, consistency is good.
            # Let's save it to a generic individual_log.txt for now or separate files.
            # Since individual mode can have 100s of calls, let's just append to one file for speed.
            log_path = os.path.join(OUTPUT_DIR, "individual_ai_responses.txt")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n--- INDIVIDUAL CALL ({time.strftime('%H:%M:%S')}) Duration: {elapsed_call:.2f}s ---\n")
                f.write(raw)
                f.write("\n" + "="*40 + "\n")

            ans_match = re.search(r'<Answer>(.*?)</Answer>', raw, re.DOTALL | re.IGNORECASE)
            answer_text = ans_match.group(1).strip() if ans_match else raw
            
            return "1" in answer_text
        except Exception as e:
            print(f"  (individual LLM error attempt {attempt+1}: {e})")
            if attempt < retries: time.sleep(2)
    return None




# ==============================
# IconCounter
# ==============================

class IconCounter:
    def __init__(self, icon_path, image_source, max_dimension=8000, scale_factors=None):
        """
        Initialize IconCounter with multi-scale scanning.
        
        Args:
            icon_path: Path to symbol image or numpy array
            image_source: Path to floor plan image or numpy array
            max_dimension: Maximum dimension for resizing
            scale_factors: List of scale factors to scan.
                          Default: [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]
        """
        self.icon_original  = cv2.imread(icon_path) if isinstance(icon_path, str) else icon_path
        self.image_original = cv2.imread(image_source) if isinstance(image_source, str) else image_source
        if self.icon_original  is None: raise ValueError(f"Could not load icon: {icon_path}")
        if self.image_original is None: raise ValueError("Could not load floor plan image.")
        self.image_height, self.image_width = self.image_original.shape[:2]
        self.icon_height,  self.icon_width  = self.icon_original.shape[:2]
        
        # Base scale factor for image resizing
        max_dim = max(self.image_width, self.image_height)
        self.base_scale_factor = 1.0
        if max_dim > max_dimension:
            self.base_scale_factor = max_dimension / max_dim
            print(f"Image {self.image_width}x{self.image_height} → downsampling {self.base_scale_factor:.2f}x")
            self.image = cv2.resize(self.image_original, None,
                                    fx=self.base_scale_factor, fy=self.base_scale_factor,
                                    interpolation=cv2.INTER_AREA)
            self.icon  = cv2.resize(self.icon_original, None,
                                    fx=self.base_scale_factor, fy=self.base_scale_factor,
                                    interpolation=cv2.INTER_AREA)
        else:
            self.image = self.image_original
            self.icon  = self.icon_original
        
        # Fixed scale factors for multi-scale scanning (default range)
        self.scale_factors = scale_factors if scale_factors is not None else [1.0]
        
        print(f"Processing at: {self.image.shape[1]}x{self.image.shape[0]}")
        print(f"Icon size:     {self.icon.shape[1]}x{self.icon.shape[0]}")
        print(f"Multi-scale scan: {len(self.scale_factors)} scales [{min(self.scale_factors):.2f}x - {max(self.scale_factors):.2f}x]")

    def rotate_image(self, image, angle):
        h,w = image.shape[:2]
        M   = cv2.getRotationMatrix2D((w//2,h//2), angle, 1.0)
        cos,sin = abs(M[0,0]),abs(M[0,1])
        nw = int(h*sin+w*cos); nh = int(h*cos+w*sin)
        M[0,2] += nw/2-w//2; M[1,2] += nh/2-h//2
        return cv2.warpAffine(image, M, (nw,nh),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))

    def _precompute_valid_tiles(self, image, is_tiled, tile_size=4000, overlap=200):
        """Pre-calculates which tiles (or the whole image) contain red pixels to skip processing empty areas."""
        if not is_tiled:
            # For _match_angle, we just check if the whole image has ANY red
            mask = _get_red_mask(image)
            return cv2.countNonZero(mask) > 10

        h, w = image.shape[:2]
        stride = tile_size - overlap
        valid_indices = set()
        
        # Pre-detect red in the whole image once
        red_mask = _get_red_mask(image)
        
        for ty in range(math.ceil((h - overlap) / stride)):
            for tx in range(math.ceil((w - overlap) / stride)):
                ys, xs = ty * stride, tx * stride
                ye, xe = min(ys + tile_size, h), min(xs + tile_size, w)
                tile_mask = red_mask[ys:ye, xs:xe]
                if cv2.countNonZero(tile_mask) > 10:
                    valid_indices.add((ty, tx))
        return valid_indices

    def _nms_advanced(self, matches, overlap_thresh=0.4):
        if not matches: return []
        matches = sorted(matches, key=lambda x: x['confidence'], reverse=True)
        filtered = []
        for m in matches:
            x1,y1=m['x'],m['y']; w,h=m['w'],m['h']
            x2,y2=x1+w,y1+h; cx,cy=x1+w/2,y1+h/2; a1=w*h; keep=True
            for k in filtered:
                kx1,ky1=k['x'],k['y']; kw,kh=k['w'],k['h']; kx2,ky2=kx1+kw,ky1+kh
                iw_=max(0,min(x2,kx2)-max(x1,kx1)); ih_=max(0,min(y2,ky2)-max(y1,ky1))
                un=a1+kw*kh-iw_*ih_; iou=(iw_*ih_)/un if un>0 else 0
                dist=math.sqrt((cx-(kx1+kw/2))**2+(cy-(ky1+kh/2))**2)
                if iou>overlap_thresh or dist<min(w,h,kw,kh)*0.4: keep=False; break
            if keep: filtered.append(m)
        return filtered

    def count_icons_robust(self, angles=None, threshold=0.6, show_matches=True,
                            n_workers=None, image_to_process=None, log_fn=None,
                            include_mirrored=False, scale_factors=None):
        """
        Count icons with optional multi-scale scanning.
        
        Args:
            angles: List of rotation angles to scan
            threshold: Confidence threshold for template matching
            show_matches: Whether to draw matches on result image
            n_workers: Number of worker threads
            image_to_process: Optional alternative image to process
            log_fn: Optional logging function
            include_mirrored: Whether to include mirrored versions of symbol
            scale_factors: Optional list of scale factors to scan (overrides instance default)
        """
        def log(msg):
            if log_fn: log_fn(msg)
            print(msg)
        if angles is None: angles=[0,45,90,135,180,225,270,315]
        if n_workers is None: n_workers = min(4, multiprocessing.cpu_count())
        
        # Use provided scale_factors or instance default
        scales_to_scan = scale_factors if scale_factors is not None else self.scale_factors
        
        max_dim    = max(self.image.shape[:2])
        worker_fn  = _match_angle_tiled if max_dim > 6000 else _match_angle
        target     = image_to_process if image_to_process is not None else self.image
        img_match  = target
        
        # ── Color Filter Initialization ───────────────────────────────────────
        valid_tile_indices = None
        if USE_COLOR_TILE_FILTER:
            if _is_red_symbol(self.icon):
                log("  Symbol detected as RED. Enabling tile color filter...")
                valid_tile_indices = self._precompute_valid_tiles(img_match, worker_fn == _match_angle_tiled)
                if isinstance(valid_tile_indices, set):
                    log(f"  Color filter: {len(valid_tile_indices)} tiles containing red.")
                else:
                    log(f"  Color filter: Whole image contains red: {valid_tile_indices}")
            else:
                log("  Symbol not predominantly red. Skipping color filter for higher accuracy.")
        # ─────────────────────────────────────────────────────────────────────

        all_matches = []
        
        # Scan across all scale factors
        for scale_idx, scale in enumerate(scales_to_scan):
            log(f"\n  🔍 Scanning at scale {scale:.2f}x ({scale_idx+1}/{len(scales_to_scan)})...")
            
            # Resize icon to current scale
            if scale == 1.0:
                scaled_icon = self.icon
            else:
                scaled_icon = cv2.resize(self.icon, None, fx=scale, fy=scale, 
                                        interpolation=cv2.INTER_AREA)
            
            rotated    = {}
            for a in angles:
                ri = self.rotate_image(scaled_icon, a)
                rotated[a] = ri

            # Also create mirrored versions if requested
            if include_mirrored:
                if scale_idx == 0:  # Only log once
                    log(f"  Including mirrored symbols (H/V flips + rotations)")
                for code, suffix in [(1, 'fH'), (0, 'fV'), (-1, 'fB')]:
                    flipped_template = cv2.flip(scaled_icon, code)
                    for a in angles:
                        ri = self.rotate_image(flipped_template, a)
                        rotated[f'{a}_{suffix}'] = ri

            tasks = [(a, rotated[a], img_match, threshold, valid_tile_indices) for a in rotated.keys()]
            log(f"  Angle scan: {len(tasks)} tasks ({n_workers} workers)...")
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                results = list(pool.map(worker_fn, tasks))
            
            scale_matches = [m for r in results for m in r]
            log(f"  Raw hits at scale {scale:.2f}x: {len(scale_matches)}")
            all_matches.extend(scale_matches)
        
        log(f"\n  📊 Total raw hits (all scales): {len(all_matches)}")
        filtered = self._nms_advanced(all_matches, overlap_thresh=0.2)
        log(f"  After NMS: {len(filtered)}")
        
        # Adjust coordinates back to original image scale
        for m in filtered:
            m['x']=int(m['x']/self.base_scale_factor); m['y']=int(m['y']/self.base_scale_factor)
            m['w']=int(m['w']/self.base_scale_factor); m['h']=int(m['h']/self.base_scale_factor)
        
        result_image = self.image_original.copy()
        if show_matches:
            for m in filtered:
                x,y,w,h=m['x'],m['y'],m['w'],m['h']
                cv2.rectangle(result_image,(x,y),(x+w,y+h),(0,0,255),3)
                cv2.putText(result_image,f"{m['confidence']:.2f}",(x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)
        return len(filtered), result_image, filtered


# ==============================
# Three-tier verification
# ==============================

def verify_all_candidates(symbol_bgr, symbol_pil, floor_pil,
                           candidates, candidate_boxes,
                           sym_analysis, log_fn=None):
    def log(msg):
        if log_fn: log_fn(msg)
        print(msg)

    total = len(candidates)
    log(f"\n🔍 Verifying {total} candidates...")
    log(f"   Tier 1 conf : auto-confirm ≥{HIGH_CONF_AUTO} | auto-reject <{LOW_CONF_AUTO}")
    log(f"   Tier 2 CV   : confirm ≥{CV_CONFIRM_SCORE} | reject <{CV_REJECT_SCORE}")

    # Warn if CV will be unreliable
    if not sym_analysis['cv_reliable']:
        log(f"\n  ⚠️  CV WARNING: {' | '.join(sym_analysis['warnings'])}")
        log(f"  ⚠️  CV scores may be inaccurate for this symbol type — LLM will handle more cases")

    cv_verifier     = CVVerifier(symbol_bgr)
    floor_np        = np.array(floor_pil)
    confirmed_boxes = []
    llm_queue       = []

    t0_cv = time.time()
    for i, (c, box) in enumerate(zip(candidates, candidate_boxes)):
        conf = c['confidence']
        if conf >= HIGH_CONF_AUTO:
            log(f"  #{i+1:>3}  ✅ T1-AUTO-CONFIRM  conf={conf:.3f}")
            confirmed_boxes.append(box); continue
        if conf < LOW_CONF_AUTO:
            log(f"  #{i+1:>3}  ❌ T1-AUTO-REJECT   conf={conf:.3f}"); continue

        x1,y1,x2,y2 = box
        crop_np  = cv2.cvtColor(floor_np[y1:y2,x1:x2], cv2.COLOR_RGB2BGR)
        # Pass the angle key from template matching to optimize CV scoring
        cv_s     = cv_verifier.score(crop_np, angle_key=c.get('angle', 0))
        cv_score = cv_s['combined']

        # If CV is known to be unreliable, widen the ambiguous zone → more goes to LLM
        eff_confirm = CV_CONFIRM_SCORE if sym_analysis['cv_reliable'] else 0.80
        eff_reject  = CV_REJECT_SCORE  if sym_analysis['cv_reliable'] else 0.10

        if cv_score >= eff_confirm:
            log(f"  #{i+1:>3}  ✅ T2-CV-CONFIRM    conf={conf:.3f}  cv={cv_score:.3f}")
            confirmed_boxes.append(box)
        elif cv_score < eff_reject:
            log(f"  #{i+1:>3}  ❌ T2-CV-REJECT     conf={conf:.3f}  cv={cv_score:.3f}")
        else:
            log(f"  #{i+1:>3}  ❓ T2-CV-AMBIGUOUS  conf={conf:.3f}  cv={cv_score:.3f}  → LLM")
            crop_pil = floor_pil.crop(box)
            llm_queue.append((i, c, box, crop_pil, cv_s))

    log(f"\n  ⏱  CV done in {time.time()-t0_cv:.2f}s | Confirmed: {len(confirmed_boxes)} | LLM queue: {len(llm_queue)}")

    # ── Tier 3: LLM ───────────────────────────────────────────────────────────
    mode = LLM_MODE  # initialise here so summary log always has access
    if llm_queue:
        log(f"\n  🤖 LLM verification ({mode} mode) for {len(llm_queue)} candidates...")

        # Save overview grid always (for your eyes)
        all_crops = [item[3] for item in llm_queue]
        ov_path   = save_grid_with_symbol(symbol_pil, all_crops, 0, OUTPUT_DIR, "_overview")
        log(f"  📷 LLM queue overview → {ov_path}")

        if mode == "batch":
            _run_batch_mode(symbol_pil, llm_queue, confirmed_boxes, log)

        elif mode == "individual":
            _run_individual_mode(symbol_pil, llm_queue, confirmed_boxes, log)

        elif mode == "both":
            log(f"\n  ⏱  Running BOTH modes to compare speed & accuracy...")
            _run_both_modes(symbol_pil, llm_queue, confirmed_boxes, log)

    # Summary
    t1_auto = sum(1 for c in candidates if c['confidence'] >= HIGH_CONF_AUTO)
    t1_rej  = sum(1 for c in candidates if c['confidence'] < LOW_CONF_AUTO)
    t2_cnt  = total - t1_auto - t1_rej
    t3_cnt  = len(llm_queue)
    log(f"\n  ══ Verification summary ══")
    log(f"  Tier 1 (instant) : {t1_auto} confirmed, {t1_rej} rejected")
    log(f"  Tier 2 (CV)      : {t2_cnt - t3_cnt} resolved without LLM")
    log(f"  Tier 3 (LLM/{mode:<10}): {t3_cnt} processed")
    log(f"  TOTAL confirmed  : {len(confirmed_boxes)} / {total}")
    return confirmed_boxes


def _run_batch_mode(symbol_pil, llm_queue, confirmed_boxes, log):
    batches = [llm_queue[i:i+LLM_BATCH_SIZE] for i in range(0, len(llm_queue), LLM_BATCH_SIZE)]
    t0 = time.time()
    for bi, batch in enumerate(batches):
        crop_pils = [item[3] for item in batch]
        grid_path = save_grid_with_symbol(symbol_pil, crop_pils, bi, OUTPUT_DIR, "_batch")
        log(f"  📷 Batch {bi+1} grid → {grid_path}")
        result_set = llm_batch(symbol_pil, crop_pils, batch_idx=bi)
        for local_idx, (global_idx, c, box, _, cv_s) in enumerate(batch, start=1):
            if result_set is None:
                log(f"  #{global_idx+1:>3}  ⚠️  T3-BATCH-ERROR")
            elif local_idx in result_set:
                log(f"  #{global_idx+1:>3}  ✅ T3-BATCH-YES    conf={c['confidence']:.3f}")
                confirmed_boxes.append(box)
            else:
                log(f"  #{global_idx+1:>3}  ❌ T3-BATCH-NO     conf={c['confidence']:.3f}")
    log(f"  ⏱  Batch mode total: {time.time()-t0:.1f}s  ({len(batches)} API call(s))")


def _run_individual_mode(symbol_pil, llm_queue, confirmed_boxes, log):
    t0 = time.time()
    for qi, (global_idx, c, box, crop_pil, cv_s) in enumerate(llm_queue):
        crop_path = os.path.join(OUTPUT_DIR, f"llm_crop_{qi+1:03d}_cand{global_idx+1}.png")
        crop_pil.save(crop_path)
        result = llm_individual(symbol_pil, crop_pil)
        elapsed = time.time()-t0
        if result is None:
            log(f"  #{global_idx+1:>3}  ⚠️  T3-IND-ERROR    conf={c['confidence']:.3f}  ({elapsed:.1f}s)")
        elif result:
            log(f"  #{global_idx+1:>3}  ✅ T3-IND-YES      conf={c['confidence']:.3f}  ({elapsed:.1f}s)")
            confirmed_boxes.append(box)
        else:
            log(f"  #{global_idx+1:>3}  ❌ T3-IND-NO       conf={c['confidence']:.3f}  ({elapsed:.1f}s)")
    log(f"  ⏱  Individual mode total: {time.time()-t0:.1f}s  ({len(llm_queue)} API calls)")


def _run_both_modes(symbol_pil, llm_queue, confirmed_boxes, log):
    """Run both modes, compare results, use batch for final decision, log timing."""
    n = len(llm_queue)

    # ── BATCH ────────────────────────────────────────────────────────────────
    log(f"\n  ── Mode A: BATCH ({math.ceil(n/LLM_BATCH_SIZE)} call(s)) ──")
    t_batch_start = time.time()
    batch_results = {}  # global_idx → bool
    batches = [llm_queue[i:i+LLM_BATCH_SIZE] for i in range(0, n, LLM_BATCH_SIZE)]
    for bi, batch in enumerate(batches):
        crop_pils = [item[3] for item in batch]
        grid_path = save_grid_with_symbol(symbol_pil, crop_pils, bi, OUTPUT_DIR, "_batch")
        log(f"  📷 Batch {bi+1} grid → {grid_path}")
        result_set = llm_batch(symbol_pil, crop_pils, batch_idx=bi)
        for local_idx, (global_idx, c, box, _, _) in enumerate(batch, start=1):
            batch_results[global_idx] = (local_idx in result_set) if result_set is not None else None
    t_batch = time.time() - t_batch_start

    # ── INDIVIDUAL ────────────────────────────────────────────────────────────
    log(f"\n  ── Mode B: INDIVIDUAL ({n} call(s)) ──")
    t_ind_start = time.time()
    ind_results = {}  # global_idx → bool
    for qi, (global_idx, c, box, crop_pil, _) in enumerate(llm_queue):
        crop_path = os.path.join(OUTPUT_DIR, f"llm_crop_{qi+1:03d}_cand{global_idx+1}.png")
        crop_pil.save(crop_path)
        result = llm_individual(symbol_pil, crop_pil)
        ind_results[global_idx] = result
        sym = "✅" if result else ("❌" if result is False else "⚠️ ")
        log(f"    #{global_idx+1:>3}  {sym}  conf={c['confidence']:.3f}")
    t_ind = time.time() - t_ind_start

    # ── Comparison report ────────────────────────────────────────────────────
    log(f"\n  ┌{'─'*56}")
    log(f"  │  SPEED COMPARISON")
    log(f"  │  Batch mode      : {t_batch:.1f}s  ({math.ceil(n/LLM_BATCH_SIZE)} API calls)")
    log(f"  │  Individual mode : {t_ind:.1f}s  ({n} API calls)")
    faster = "BATCH" if t_batch < t_ind else "INDIVIDUAL"
    log(f"  │  ✨ Faster: {faster}  ({abs(t_batch-t_ind):.1f}s difference)")
    log(f"  │")
    log(f"  │  RESULT COMPARISON")
    agreements   = 0
    disagreements= 0
    for global_idx, _, box, _, _ in [(item[0],)+item[1:] for item in llm_queue]:
        br = batch_results.get(global_idx)
        ir = ind_results.get(global_idx)
        agree = (br == ir)
        if agree: agreements += 1
        else:     disagreements += 1
        tag = "✅ AGREE" if agree else f"❗ DISAGREE (batch={'YES' if br else 'NO'}, ind={'YES' if ir else 'NO'})"
        log(f"  │  Cand #{global_idx+1:>3}: {tag}")
    log(f"  │")
    log(f"  │  Agreements   : {agreements}/{n}")
    log(f"  │  Disagreements: {disagreements}/{n}")
    log(f"  └{'─'*56}")

    # Use BATCH results as final decision (fewer API calls = production choice)
    log(f"\n  → Using BATCH results for final confirmation")
    for global_idx, c, box, _, _ in [(item[0],)+item[1:] for item in llm_queue]:
        br = batch_results.get(global_idx)
        if br:
            confirmed_boxes.append(box)


# ==============================
# MAIN
# ==============================

def count_symbol(symbol_path, floor_plan_path,
                 output_path="output_detected.png", log_fn=None,
                 scale_factors=None):
    """
    Count symbols in a floor plan with optional multi-scale scanning.
    
    Args:
        symbol_path: Path to symbol template image
        floor_plan_path: Path to floor plan image
        output_path: Path for output annotated image
        log_fn: Optional logging function
        scale_factors: List of scale factors to scan (e.g., [0.8, 1.0, 1.2]).
                      If None, only uses base scale (1.0).
    """
    start = time.time()

    global OUTPUT_DIR
    if not hasattr(count_symbol, "_base_output_dir"):
        count_symbol._base_output_dir = OUTPUT_DIR
    OUTPUT_DIR = os.path.join(count_symbol._base_output_dir, Path(symbol_path).stem)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def log(msg):
        if log_fn: log_fn(msg)
        print(msg)

    log(f"\n{'='*55}")
    log(f"  Processing : {Path(floor_plan_path).name}")
    log(f"  Symbol     : {Path(symbol_path).name}")
    log(f"  LLM mode   : {LLM_MODE}")
    log(f"  Debug dir  : {os.path.abspath(OUTPUT_DIR)}")
    log(f"{'='*55}")

    import shutil
    sym_ref_path = os.path.join(OUTPUT_DIR, "target_symbol.png")
    shutil.copy2(symbol_path, sym_ref_path)

    symbol_pil = Image.open(symbol_path).convert("RGB")
    symbol_bgr = cv2.imread(symbol_path)
    floor_w, floor_h = Image.open(floor_plan_path).size

    # ── Symbol analysis ───────────────────────────────────────────────────────
    log("\n[Step 0] Analyzing symbol...")
    sym_analysis = analyze_symbol(symbol_bgr)
    log(f"  Simple rectangle : {sym_analysis['is_simple_rect']}")
    log(f"  Yellow color     : {sym_analysis['is_yellow']}")
    log(f"  ORB keypoints    : {sym_analysis['orb_keypoints']}")
    log(f"  CV reliable      : {sym_analysis['cv_reliable']}")
    if sym_analysis['warnings']:
        for w in sym_analysis['warnings']:
            log(f"  {w}")

    # ── HIGHLIGHT COLORS ───────────────────────────────────────────────────
    log("\n[Step 1] Color highlighting (from symbol top colors)...")
    
    top_hues = extract_top_hsv_colors(symbol_bgr, num_colors=2)
    if not top_hues:
        # Fallback to no highlight if image is perfectly grey/black/white
        top_hues = []
    
    color_range_pct = 0.10 # 10% hue tolerance
    log(f"  Extracted top Hue(s) from symbol: {top_hues}")

    floor_cv = cv2.imread(floor_plan_path)

    log(f"  Applying highlight (boost to RED, make background BLACK, {color_range_pct*100:.0f}% tolerance) on symbol and floor plan...")
    highlighted_floor = highlight_hsv_colors(floor_cv, top_hues, color_range_pct=color_range_pct, boost_to=(0, 0, 255), darken_factor=0.0)
    highlighted_symbol = highlight_hsv_colors(symbol_bgr, top_hues, color_range_pct=color_range_pct, boost_to=(0, 0, 255), darken_factor=0.0)
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, "debug_highlighted_floor.png"), highlighted_floor)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "debug_highlighted_symbol.png"), highlighted_symbol)


    # Save the highlighted symbol and a known crop from the floor plan side-by-side
    known_x, known_y = 100, 100  # coordinates where you know the symbol exists
    sh, sw = highlighted_symbol.shape[:2]
    debug_crop = highlighted_floor[known_y:known_y+sh, known_x:known_x+sw]
    cv2.imwrite(os.path.join(OUTPUT_DIR, "debug_symbol_vs_crop.png"),
                np.hstack([highlighted_symbol, debug_crop]))

    # ── Template matching ─────────────────────────────────────────────────────
    log("\n[Step 2] Template matching...")
    counter = IconCounter(highlighted_symbol, highlighted_floor, max_dimension=8000, 
                         scale_factors=scale_factors)

    log("  Scan: highlighted image (with mirrored symbols)")
    _, _, matches_filt = counter.count_icons_robust(
        angles = list(range(0, 360, 45)), threshold=0.6,
        show_matches=False,
        n_workers=min(4, multiprocessing.cpu_count()), include_mirrored=True, log_fn=log_fn)

    final_matches = counter._nms_advanced(matches_filt, overlap_thresh=0.2)
    n_candidates  = len(final_matches)
    log(f"  Final unique candidates: {n_candidates}")

    # ── ⚠️ CANDIDATE WARNING ─────────────────────────────────────────────────
    if n_candidates > CANDIDATE_WARNING_LIMIT:
        log(f"\n{'!'*55}")
        log(f"  ⚠️  WARNING: {n_candidates} candidates found!")
        log(f"  ⚠️  This is above the limit of {CANDIDATE_WARNING_LIMIT}.")
        log(f"  ⚠️  Possible causes:")
        log(f"       • Symbol is too generic (simple shape, common color)")
        log(f"       • Template matching threshold too low (currently 0.5)")
        log(f"       • Symbol color is very common on this floor plan")
        log(f"  ⚠️  RECOMMENDATION: Please inspect 'candidates_overview.png'")
        log(f"       and verify manually. The automatic count may be unreliable.")
        log(f"  ⚠️  SCAN CANCELLED — returning 0")
        log(f"{'!'*55}")

        # Still save candidates overview so user can inspect
        floor_pil_ann = Image.open(floor_plan_path).convert("RGB")
        draw          = ImageDraw.Draw(floor_pil_ann)
        try:    lf = ImageFont.truetype("arial.ttf", max(20,int(min(floor_w,floor_h)/120)))
        except: lf = None
        for idx, m in enumerate(final_matches):
            x1=max(0,m['x']-CROP_PADDING);   y1=max(0,m['y']-CROP_PADDING)
            x2=min(floor_w,m['x']+m['w']+CROP_PADDING)
            y2=min(floor_h,m['y']+m['h']+CROP_PADDING)
            draw.rectangle((x1,y1,x2,y2), outline="orange", width=2)
        cand_path = os.path.join(OUTPUT_DIR, "candidates_overview_WARNING.png")
        floor_pil_ann.save(cand_path)
        log(f"  📋 Candidates saved for manual inspection → {cand_path}")
        return -1   # -1 signals "cancelled due to too many candidates"

    candidates = [{'x':m['x'],'y':m['y'],'w':m['w'],'h':m['h'],
                   'cx':m['x']+m['w']//2,'cy':m['y']+m['h']//2,
                   'confidence':m['confidence']} for m in final_matches]

    if not candidates:
        log("No candidates found."); return 0

    # ── Build crops ───────────────────────────────────────────────────────────
    floor_pil_clean = Image.open(floor_plan_path).convert("RGB")
    floor_pil_annot = floor_pil_clean.copy()
    draw            = ImageDraw.Draw(floor_pil_annot)
    candidate_boxes = []
    try:    _fs = max(20,int(min(floor_w,floor_h)/120)); _lf = ImageFont.truetype("arial.ttf",_fs)
    except: _lf=None; _fs=12

    for idx, c in enumerate(candidates):
        x1=max(0,c['x']-CROP_PADDING);  y1=max(0,c['y']-CROP_PADDING)
        x2=min(floor_w,c['x']+c['w']+CROP_PADDING)
        y2=min(floor_h,c['y']+c['h']+CROP_PADDING)
        box=(x1,y1,x2,y2); candidate_boxes.append(box)
        draw.rectangle(box, outline="blue", width=2)
        lbl=str(idx+1); tx,ty=x1+3,y1+3
        draw.rectangle((tx-2,ty-2,tx+_fs*len(lbl)//2+4,ty+_fs+2), fill="blue")
        draw.text((tx,ty), lbl, fill="white", font=_lf)

    cand_path = os.path.join(OUTPUT_DIR, "candidates_overview.png")
    floor_pil_annot.save(cand_path)
    log(f"\n📋 Candidates overview → {cand_path}")

    # ── Verification ──────────────────────────────────────────────────────────
    log("\n[Step 3] Three-tier verification...")
    confirmed = verify_all_candidates(
        symbol_bgr, symbol_pil, floor_pil_clean,
        candidates, candidate_boxes, sym_analysis, log_fn=log_fn)

    # ── Final annotated output ────────────────────────────────────────────────
    for box in confirmed:
        draw.rectangle(box, outline="red", width=10)
        oi  = candidate_boxes.index(box); lbl=str(oi+1)
        tx,ty = box[0]+3,box[1]+3
        draw.rectangle((tx-2,ty-2,tx+_fs*len(lbl)//2+4,ty+_fs+2), fill="red")
        draw.text((tx,ty), lbl, fill="white", font=_lf)

    floor_pil_annot.save(output_path)
    elapsed = time.time()-start
    log(f"\n{'='*55}")
    log(f"  Output     : {Path(output_path).name}")
    log(f"  Elapsed    : {elapsed:.1f}s")
    log(f"  Candidates : {len(candidates)}")
    log(f"  Confirmed  : {len(confirmed)}")
    log(f"{'='*55}")
    return len(confirmed)


if __name__ == "__main__":
    start_time = time.perf_counter()  # High-resolution timer start
    multiprocessing.freeze_support()
    
    # Example: Multi-scale scanning (uncomment to use)
    # scale_factors = [0.8, 0.9, 1.0, 1.1, 1.2]  # Scan at different scales

    
    result = count_symbol(
        symbol_path     = SYMBOL_PATH,
        floor_plan_path = FLOOR_PLAN_PATH,
        output_path     = os.path.join(OUTPUT_DIR, "output_detected.png"),
        # scale_factors = scale_factors  # Uncomment to enable multi-scale scan
    )
    if result == -1:
        print("\n⛔ Scan cancelled — too many candidates. Please check candidates_overview_WARNING.png")
    else:
        print(f"\n✅ Final count: {result} symbol(s) found")
    print(f"📁 Debug images: {os.path.abspath(OUTPUT_DIR)}/")
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"\n⏱️  Total execution time: {total_time:.2f} seconds")