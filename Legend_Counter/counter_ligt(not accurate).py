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
CROP_PADDING    = 55

# ── Candidate warning threshold ───────────────────────────────────────────────
CANDIDATE_WARNING_LIMIT = 300

# ── LLM Mode ─────────────────────────────────────────────────────────────────
LLM_MODE       = "batch"
LLM_BATCH_SIZE = 3

# ── Confidence thresholds ─────────────────────────────────────────────────────
HIGH_CONF_AUTO   = 0.93
LOW_CONF_AUTO    = 0.60
CV_CONFIRM_SCORE = 0.55
CV_REJECT_SCORE  = 0.30
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
# Symbol analysis helpers
# ==============================

def analyze_symbol(symbol_bgr: np.ndarray) -> dict:
    gray   = cv2.cvtColor(symbol_bgr, cv2.COLOR_BGR2GRAY)
    h, w   = gray.shape[:2]
    warnings = []

    # ── Check if symbol is a simple rectangle ────────────────────────────────
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

    # ── ORB keypoint count ────────────────────────────────────────────────────
    orb     = cv2.ORB_create(nfeatures=500)
    kp, _   = orb.detectAndCompute(gray, None)
    n_kp    = len(kp) if kp else 0
    if n_kp < 10:
        warnings.append(f"⚠️  Only {n_kp} ORB keypoints detected — ORB verification will score ~0.00")

    return {
        'is_simple_rect': is_simple_rect,
        'orb_keypoints':  n_kp,
        'warnings':       warnings,
        'cv_reliable':    not is_simple_rect and n_kp >= 10,
    }


# ==============================
# Template matching workers
# ==============================

def _match_angle(task):
    angle, icon, image, threshold = task
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


def _match_angle_tiled(task, tile_size=4000, overlap=200):
    angle, icon, image, threshold = task
    ih, iw  = icon.shape[:2]
    h, w    = image.shape[:2]
    matches = []
    stride  = tile_size - overlap
    for ty in range(math.ceil((h - overlap) / stride)):
        for tx in range(math.ceil((w - overlap) / stride)):
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
    print(f"  Angle {angle:>6}: {len(matches)} raw hits (tiled)")
    return matches


# ==============================
# CV Verifier
# ==============================

class CVVerifier:
    ANGLES = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5]

    def __init__(self, symbol_bgr):
        self.symbol_gray = cv2.cvtColor(symbol_bgr, cv2.COLOR_BGR2GRAY)
        self.symbol_bgr  = symbol_bgr
        self._rotated    = {a: self._rotate(self.symbol_gray, a) for a in self.ANGLES}
        self.orb         = cv2.ORB_create(nfeatures=500)
        self.bf          = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        sym_kp, sym_des  = self.orb.detectAndCompute(self.symbol_gray, None)
        self.sym_kp      = sym_kp
        self.sym_des     = sym_des
        self.sym_hist    = self._hist(symbol_bgr)

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

    def _ssim_score(self, crop_gray):
        best = 0.0
        for a, sym_rot in self._rotated.items():
            ch, cw = crop_gray.shape[:2]
            sh, sw = sym_rot.shape[:2]
            if sh==0 or sw==0: continue
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
        try:
            matches = self.bf.match(self.sym_des, des2)
            good    = [m for m in matches if m.distance < 50]
            return min(1.0, len(good) / max(len(self.sym_kp), 1))
        except: return 0.0

    def _hist_score(self, crop_bgr):
        ch = self._hist(crop_bgr)
        return float(cv2.compareHist(
            self.sym_hist.reshape(-1,1).astype(np.float32),
            ch.reshape(-1,1).astype(np.float32),
            cv2.HISTCMP_CORREL))

    def score(self, crop_bgr):
        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        ssim_s    = self._ssim_score(crop_gray)
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
    CELL=256; GAP=4; PAD=12
    sw,sh  = symbol_pil.size
    sym_up = symbol_pil.resize((sw*3,sh*3), Image.BICUBIC)
    sym_up.thumbnail((CELL*2,CELL*2), Image.LANCZOS)
    grid = _make_grid(crop_pils, cols=min(3,len(crop_pils)), cell_size=CELL, gap=GAP)
    total_w = sym_up.width + PAD*3 + grid.width
    total_h = max(sym_up.height, grid.height) + PAD*2 + 40
    canvas  = Image.new("RGB", (total_w, total_h), (245,245,245))
    draw    = ImageDraw.Draw(canvas)
    try:    tf = ImageFont.truetype("arial.ttf", 18)
    except: tf = None
    title = f"Batch {batch_idx+1}  {label}  |  TARGET (left)  CROPS (right)"
    draw.text((PAD, 6), title, fill=(30,30,30), font=tf)
    draw.rectangle((PAD-2, 40-2, PAD+sym_up.width+2, 40+sym_up.height+2),
                   outline=(220,80,80), width=4)
    canvas.paste(sym_up, (PAD, 40))
    canvas.paste(grid, (PAD*2+sym_up.width, 40))
    return canvas

def save_grid_with_symbol(symbol_pil, crop_pils, batch_idx, out_dir, label=""):
    canvas = create_combined_image(symbol_pil, crop_pils, batch_idx, label)
    path = os.path.join(out_dir, f"batch_{batch_idx+1:03d}_grid{label}.png")
    canvas.save(path)
    return path


# ==============================
# LLM — BATCH mode
# ==============================

def llm_batch(symbol_pil, crop_pils, batch_idx=0, retries=2):
    n = len(crop_pils)
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
            try:
                log_path = os.path.join(OUTPUT_DIR, f"batch_{batch_idx+1:03d}_error.txt")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"Error at {time.strftime('%H:%M:%S')}: {str(e)}\n")
            except: pass
            print(f"  (batch LLM error attempt {attempt+1}: {e})")
            if attempt < retries: time.sleep(2)
    return None


# ==============================
# LLM — INDIVIDUAL mode
# ==============================

def llm_individual(symbol_pil, crop_pil, retries=2):
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
    def __init__(self, icon_path, image_source):
        self.icon_original  = cv2.imread(icon_path)
        self.image_original = cv2.imread(image_source) if isinstance(image_source, str) else image_source
        if self.icon_original  is None: raise ValueError(f"Could not load icon: {icon_path}")
        if self.image_original is None: raise ValueError("Could not load floor plan image.")
        self.image_height, self.image_width = self.image_original.shape[:2]
        self.icon_height,  self.icon_width  = self.icon_original.shape[:2]
        # No downsampling — use images as-is
        self.scale_factor = 1.0
        self.image = self.image_original
        self.icon  = self.icon_original
        print(f"Processing at: {self.image.shape[1]}x{self.image.shape[0]}")
        print(f"Icon size:     {self.icon.shape[1]}x{self.icon.shape[0]}")

    def rotate_image(self, image, angle):
        h,w = image.shape[:2]
        M   = cv2.getRotationMatrix2D((w//2,h//2), angle, 1.0)
        cos,sin = abs(M[0,0]),abs(M[0,1])
        nw = int(h*sin+w*cos); nh = int(h*cos+w*sin)
        M[0,2] += nw/2-w//2; M[1,2] += nh/2-h//2
        return cv2.warpAffine(image, M, (nw,nh),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))

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

    def count_icons_robust(self, angles=None, threshold=0.5, show_matches=True,
                            n_workers=None, image_to_process=None, log_fn=None):
        def log(msg):
            if log_fn: log_fn(msg)
            print(msg)
        if angles is None: angles=[0,45,90,135,180,225,270,315]
        if n_workers is None: n_workers=multiprocessing.cpu_count()
        max_dim   = max(self.image.shape[:2])
        worker_fn = _match_angle_tiled if max_dim > 6000 else _match_angle
        target    = image_to_process if image_to_process is not None else self.image

        # Use color images directly — no greyscale conversion
        rotated = {a: self.rotate_image(self.icon, a) for a in angles}
        tasks   = [(a, rotated[a], target, threshold) for a in angles]

        log(f"  Angle scan: {len(tasks)} tasks ({n_workers} workers)...")
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(worker_fn, tasks))
        all_matches = [m for r in results for m in r]
        log(f"  Raw hits: {len(all_matches)}")
        filtered = self._nms_advanced(all_matches, overlap_thresh=0.2)
        log(f"  After NMS: {len(filtered)}")
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
        cv_s     = cv_verifier.score(crop_np)
        cv_score = cv_s['combined']

        eff_confirm = CV_CONFIRM_SCORE if sym_analysis['cv_reliable'] else 0.80
        eff_reject  = CV_REJECT_SCORE  if sym_analysis['cv_reliable'] else 0.30

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

    mode = LLM_MODE
    if llm_queue:
        log(f"\n  🤖 LLM verification ({mode} mode) for {len(llm_queue)} candidates...")
        all_crops = [item[3] for item in llm_queue]
        ov_path   = save_grid_with_symbol(symbol_pil, all_crops, 0, OUTPUT_DIR, "_overview")
        log(f"  📷 LLM queue overview → {ov_path}")

        if mode == "batch":
            _run_batch_mode(symbol_pil, llm_queue, confirmed_boxes, log)
        elif mode == "individual":
            _run_individual_mode(symbol_pil, llm_queue, confirmed_boxes, log)
        elif mode == "both":
            _run_both_modes(symbol_pil, llm_queue, confirmed_boxes, log)

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
    n = len(llm_queue)

    log(f"\n  ── Mode A: BATCH ({math.ceil(n/LLM_BATCH_SIZE)} call(s)) ──")
    t_batch_start = time.time()
    batch_results = {}
    batches = [llm_queue[i:i+LLM_BATCH_SIZE] for i in range(0, n, LLM_BATCH_SIZE)]
    for bi, batch in enumerate(batches):
        crop_pils = [item[3] for item in batch]
        grid_path = save_grid_with_symbol(symbol_pil, crop_pils, bi, OUTPUT_DIR, "_batch")
        log(f"  📷 Batch {bi+1} grid → {grid_path}")
        result_set = llm_batch(symbol_pil, crop_pils, batch_idx=bi)
        for local_idx, (global_idx, c, box, _, _) in enumerate(batch, start=1):
            batch_results[global_idx] = (local_idx in result_set) if result_set is not None else None
    t_batch = time.time() - t_batch_start

    log(f"\n  ── Mode B: INDIVIDUAL ({n} call(s)) ──")
    t_ind_start = time.time()
    ind_results = {}
    for qi, (global_idx, c, box, crop_pil, _) in enumerate(llm_queue):
        crop_path = os.path.join(OUTPUT_DIR, f"llm_crop_{qi+1:03d}_cand{global_idx+1}.png")
        crop_pil.save(crop_path)
        result = llm_individual(symbol_pil, crop_pil)
        ind_results[global_idx] = result
        sym = "✅" if result else ("❌" if result is False else "⚠️ ")
        log(f"    #{global_idx+1:>3}  {sym}  conf={c['confidence']:.3f}")
    t_ind = time.time() - t_ind_start

    log(f"\n  ┌{'─'*56}")
    log(f"  │  SPEED COMPARISON")
    log(f"  │  Batch mode      : {t_batch:.1f}s  ({math.ceil(n/LLM_BATCH_SIZE)} API calls)")
    log(f"  │  Individual mode : {t_ind:.1f}s  ({n} API calls)")
    faster = "BATCH" if t_batch < t_ind else "INDIVIDUAL"
    log(f"  │  ✨ Faster: {faster}  ({abs(t_batch-t_ind):.1f}s difference)")
    log(f"  │")
    log(f"  │  RESULT COMPARISON")
    agreements = disagreements = 0
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

    log(f"\n  → Using BATCH results for final confirmation")
    for global_idx, c, box, _, _ in [(item[0],)+item[1:] for item in llm_queue]:
        br = batch_results.get(global_idx)
        if br:
            confirmed_boxes.append(box)


# ==============================
# MAIN
# ==============================

def count_symbol(symbol_path, floor_plan_path,
                 output_path="output_detected.png", log_fn=None):
    start = time.time()

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
    log(f"  ORB keypoints    : {sym_analysis['orb_keypoints']}")
    log(f"  CV reliable      : {sym_analysis['cv_reliable']}")
    if sym_analysis['warnings']:
        for w in sym_analysis['warnings']:
            log(f"  {w}")

    # ── Template matching ─────────────────────────────────────────────────────
    log("\n[Step 1] Template matching on original color image...")
    counter = IconCounter(symbol_path, floor_plan_path)

    _, _, matches = counter.count_icons_robust(
        angles=[0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5,
                180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5],
        threshold=0.5, show_matches=False,
        n_workers=multiprocessing.cpu_count(), log_fn=log_fn)

    n_candidates = len(matches)
    log(f"  Final unique candidates: {n_candidates}")

    # ── ⚠️ CANDIDATE WARNING ─────────────────────────────────────────────────
    if n_candidates > CANDIDATE_WARNING_LIMIT:
        log(f"\n{'!'*55}")
        log(f"  ⚠️  WARNING: {n_candidates} candidates found!")
        log(f"  ⚠️  This is above the limit of {CANDIDATE_WARNING_LIMIT}.")
        log(f"  ⚠️  Possible causes:")
        log(f"       • Symbol is too generic (simple shape, common color)")
        log(f"       • Template matching threshold too low (currently 0.5)")
        log(f"  ⚠️  RECOMMENDATION: Please inspect 'candidates_overview.png'")
        log(f"       and verify manually. The automatic count may be unreliable.")
        log(f"  ⚠️  SCAN CANCELLED — returning 0")
        log(f"{'!'*55}")

        floor_pil_ann = Image.open(floor_plan_path).convert("RGB")
        draw          = ImageDraw.Draw(floor_pil_ann)
        try:    lf = ImageFont.truetype("arial.ttf", max(20,int(min(floor_w,floor_h)/120)))
        except: lf = None
        for idx, m in enumerate(matches):
            x1=max(0,m['x']-CROP_PADDING);   y1=max(0,m['y']-CROP_PADDING)
            x2=min(floor_w,m['x']+m['w']+CROP_PADDING)
            y2=min(floor_h,m['y']+m['h']+CROP_PADDING)
            draw.rectangle((x1,y1,x2,y2), outline="orange", width=2)
        cand_path = os.path.join(OUTPUT_DIR, "candidates_overview_WARNING.png")
        floor_pil_ann.save(cand_path)
        log(f"  📋 Candidates saved for manual inspection → {cand_path}")
        return -1

    candidates = [{'x':m['x'],'y':m['y'],'w':m['w'],'h':m['h'],
                   'cx':m['x']+m['w']//2,'cy':m['y']+m['h']//2,
                   'confidence':m['confidence']} for m in matches]

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
    log("\n[Step 2] Three-tier verification...")
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
    multiprocessing.freeze_support()
    result = count_symbol(
        symbol_path     = SYMBOL_PATH,
        floor_plan_path = FLOOR_PLAN_PATH,
        output_path     = os.path.join(OUTPUT_DIR, "output_detected.png")
    )
    if result == -1:
        print("\n⛔ Scan cancelled — too many candidates. Please check candidates_overview_WARNING.png")
    else:
        print(f"\n✅ Final count: {result} symbol(s) found")
    print(f"📁 Debug images: {os.path.abspath(OUTPUT_DIR)}/")