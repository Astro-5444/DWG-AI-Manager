import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests, base64, time, math, multiprocessing
from io import BytesIO
from typing import Tuple, List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage.metrics import structural_similarity as ssim

Image.MAX_IMAGE_PIXELS = 500000000

# ==============================
# CONFIGURATION
# ==============================
API_KEY    = "ak_OFIKwiNCW2UcDWkLrVRhMR-tVb9SIwaGvGeGueDk1tM"
BASE_URL   = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "qwen-3vl"

SYMBOL     = r"D:\AVIS\Manager\output\small_test\symbols\symbol_4.png"
FLOOR_PLAN = r"D:\AVIS\Manager\output\small_test\43001-AJI-04-DWG-IC-BL1-210004-000\floor_plan.png"


CROP_PADDING = 55

# ── Speed Knobs ────────────────────────────────────────────────────────────────
#
# CONFIDENCE THRESHOLDS (template matching score 0–1)
HIGH_CONF_AUTO   = 0.93   # auto-confirm, no CV or LLM needed
LOW_CONF_AUTO    = 0.6   # auto-reject, no CV or LLM needed
#
# CV VERIFICATION (runs in milliseconds — replaces most LLM calls)
CV_CONFIRM_SCORE = 0.55   # combined CV score → confirm, skip LLM
CV_REJECT_SCORE  = 0.30   # combined CV score → reject, skip LLM
# Between the two thresholds → LLM fallback (should be very rare)
#
# LLM (last resort only)
LLM_BATCH_SIZE   = 6      # crops per single LLM call
# ──────────────────────────────────────────────────────────────────────────────


# ==============================
# Color filtering helpers
# ==============================

def extract_icon_colors(icon_img, brightness_threshold=240):
    if icon_img is None:
        return []
    pixels = icon_img.reshape(-1, 3)
    mask = np.any(pixels < brightness_threshold, axis=1)
    fg_pixels = pixels[mask]
    if len(fg_pixels) == 0:
        return []
    quantized = (fg_pixels.astype(np.int32) // 16) * 16
    unique_colors = np.unique(quantized, axis=0).astype(np.uint8)
    return unique_colors


def filter_by_color(image, icon_colors, tolerance=30):
    if len(icon_colors) == 0:
        print("No colors detected in icon — skipping color filter.")
        return image.copy()
    h, w = image.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for color in icon_colors:
        lower = np.clip(color.astype(np.int32) - tolerance,      0, 255).astype(np.uint8)
        upper = np.clip(color.astype(np.int32) + 15 + tolerance, 0, 255).astype(np.uint8)
        mask = cv2.inRange(image, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    result = np.full_like(image, 255)
    result[combined_mask > 0] = image[combined_mask > 0]
    return result


# ==============================
# Template matching workers
# ==============================

def _match_angle(task):
    angle, icon, image, threshold = task
    ih, iw = icon.shape[:2]
    matches = []
    if ih > image.shape[0] or iw > image.shape[1]:
        return matches
    result = cv2.matchTemplate(image, icon, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    for pt in zip(*locations[::-1]):
        matches.append({
            'x': int(pt[0]), 'y': int(pt[1]),
            'w': iw, 'h': ih,
            'angle': angle,
            'confidence': float(result[pt[1], pt[0]])
        })
    print(f"  Angle {angle:>6}: {len(matches)} raw hits")
    return matches


def _match_angle_tiled(task, tile_size: int = 4000, overlap: int = 200):
    angle, icon, image, threshold = task
    ih, iw = icon.shape[:2]
    h, w = image.shape[:2]
    matches = []
    stride = tile_size - overlap
    n_tiles_y = math.ceil((h - overlap) / stride)
    n_tiles_x = math.ceil((w - overlap) / stride)
    for tile_y in range(n_tiles_y):
        for tile_x in range(n_tiles_x):
            y_start = tile_y * stride
            x_start = tile_x * stride
            y_end = min(y_start + tile_size, h)
            x_end = min(x_start + tile_size, w)
            tile = image[y_start:y_end, x_start:x_end]
            if ih > tile.shape[0] or iw > tile.shape[1]:
                continue
            result = cv2.matchTemplate(tile, icon, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)
            for pt in zip(*locations[::-1]):
                matches.append({
                    'x': int(pt[0]) + x_start,
                    'y': int(pt[1]) + y_start,
                    'w': iw, 'h': ih,
                    'angle': angle,
                    'confidence': float(result[pt[1], pt[0]])
                })
    print(f"  Angle {angle:>6}: {len(matches)} raw hits (tiled)")
    return matches


# ==============================
# CV Verification (fast, no LLM)
# ==============================

class CVVerifier:
    """
    Verifies a floor-plan crop against a symbol using three fast CV methods.
    Returns a combined score 0–1. Runs in <10ms per candidate.
    """

    ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]

    def __init__(self, symbol_bgr: np.ndarray):
        self.symbol_gray = cv2.cvtColor(symbol_bgr, cv2.COLOR_BGR2GRAY)
        self.symbol_bgr  = symbol_bgr

        # Pre-compute rotated symbol patches for SSIM
        self._rotated_gray = {}
        for a in self.ANGLES:
            self._rotated_gray[a] = self._rotate(self.symbol_gray, a)

        # ORB for feature matching
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        sym_kp, sym_des = self.orb.detectAndCompute(self.symbol_gray, None)
        self.sym_kp  = sym_kp
        self.sym_des = sym_des

        # Symbol histogram
        self.sym_hist = self._hist(symbol_bgr)

    @staticmethod
    def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
        h, w = img.shape[:2]
        M   = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw = int(h * sin + w * cos)
        nh = int(h * cos + w * sin)
        M[0, 2] += nw / 2 - w // 2
        M[1, 2] += nh / 2 - h // 2
        return cv2.warpAffine(img, M, (nw, nh),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    @staticmethod
    def _hist(bgr: np.ndarray) -> np.ndarray:
        h = cv2.calcHist([bgr], [0, 1, 2], None, [8, 8, 8], [0,256,0,256,0,256])
        cv2.normalize(h, h)
        return h.flatten()

    def _ssim_score(self, crop_gray: np.ndarray) -> float:
        """Best SSIM across all rotations of the symbol."""
        best = 0.0
        for a, sym_rot in self._rotated_gray.items():
            # Resize symbol rotation to match crop size
            ch, cw = crop_gray.shape[:2]
            sh, sw = sym_rot.shape[:2]
            if sh == 0 or sw == 0:
                continue
            scale = min(cw / sw, ch / sh)
            if scale <= 0:
                continue
            resized = cv2.resize(sym_rot, (max(1, int(sw * scale)), max(1, int(sh * scale))),
                                 interpolation=cv2.INTER_AREA)
            rh, rw = resized.shape[:2]
            if rh > ch or rw > cw:
                continue
            # Centre-crop region of same size
            y_off = (ch - rh) // 2
            x_off = (cw - rw) // 2
            region = crop_gray[y_off:y_off + rh, x_off:x_off + rw]
            if region.shape != resized.shape or min(rh, rw) < 7:
                continue
            try:
                score, _ = ssim(region, resized, full=True)
                best = max(best, score)
            except Exception:
                pass
        return max(0.0, best)

    def _orb_score(self, crop_gray: np.ndarray) -> float:
        """ORB keypoint match ratio."""
        if self.sym_des is None:
            return 0.0
        kp2, des2 = self.orb.detectAndCompute(crop_gray, None)
        if des2 is None or len(kp2) == 0:
            return 0.0
        try:
            matches = self.bf.match(self.sym_des, des2)
            if len(matches) == 0:
                return 0.0
            # Score = good matches / expected matches (capped at 1)
            good = [m for m in matches if m.distance < 50]
            return min(1.0, len(good) / max(len(self.sym_kp), 1))
        except Exception:
            return 0.0

    def _hist_score(self, crop_bgr: np.ndarray) -> float:
        """Histogram correlation."""
        crop_hist = self._hist(crop_bgr)
        return float(cv2.compareHist(
            self.sym_hist.reshape(-1, 1).astype(np.float32),
            crop_hist.reshape(-1, 1).astype(np.float32),
            cv2.HISTCMP_CORREL
        ))

    def score(self, crop_bgr: np.ndarray) -> dict:
        """
        Returns dict with individual and combined scores (all 0–1).
        Combined = weighted average: SSIM 50% | ORB 30% | Hist 20%
        """
        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        ssim_s = self._ssim_score(crop_gray)
        orb_s  = self._orb_score(crop_gray)
        hist_s = max(0.0, self._hist_score(crop_bgr))
        combined = ssim_s * 0.50 + orb_s * 0.30 + hist_s * 0.20
        return {
            'ssim': ssim_s,
            'orb':  orb_s,
            'hist': hist_s,
            'combined': combined,
        }


# ==============================
# LLM helpers (last-resort only)
# ==============================

def image_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_grid(crops: List[Image.Image], cols: int = 3, cell_size: int = 256, gap: int = 4) -> Image.Image:
    rows   = math.ceil(len(crops) / cols)
    grid_w = cols * cell_size + (cols - 1) * gap
    grid_h = rows * cell_size + (rows - 1) * gap
    grid   = Image.new("RGB", (grid_w, grid_h), (220, 220, 220))
    draw   = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font = None
    for i, crop in enumerate(crops):
        col   = i % cols
        row   = i // cols
        x_off = col * (cell_size + gap)
        y_off = row * (cell_size + gap)
        thumb = crop.copy()
        thumb.thumbnail((cell_size - 4, cell_size - 4), Image.LANCZOS)
        tw, th = thumb.size
        grid.paste(thumb, (x_off + (cell_size - tw) // 2, y_off + (cell_size - th) // 2))
        draw.rectangle((x_off, y_off, x_off + cell_size - 1, y_off + cell_size - 1),
                       outline=(80, 80, 200), width=2)
        draw.rectangle((x_off, y_off, x_off + 28, y_off + 26), fill=(80, 80, 200))
        draw.text((x_off + 4, y_off + 2), str(i + 1), fill="white", font=font)
    return grid


def verify_batch_llm(symbol_pil: Image.Image, crop_pils: List[Image.Image], retries: int = 2) -> Optional[set]:
    """Send a grid of crops to LLM. Returns set of 1-based confirmed indices, or None on error."""
    n = len(crop_pils)

    # Upscale symbol 3x for clarity
    sw, sh = symbol_pil.size
    sym_up = symbol_pil.resize((sw * 3, sh * 3), Image.BICUBIC)
    grid   = _make_grid(crop_pils, cols=min(3, n))

    sym_b64  = image_to_base64(sym_up)
    grid_b64 = image_to_base64(grid)
    idx_list = ", ".join(str(i + 1) for i in range(n))

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": (
                "You are a precise symbol-matching assistant. "
                "Given a TARGET SYMBOL and a numbered grid of floor-plan crops, "
                "list which cell numbers contain the target symbol (rotated versions count). "
                "Some Symbols might be connected with wires or under wires or text/other opject,So check that with yes."
                "Reply ONLY with comma-separated numbers or NONE."
            )},
            {"role": "user", "content": [
                {"type": "text",      "text": "TARGET SYMBOL:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{sym_b64}"}},
                {"type": "text",      "text": f"\nGRID (cells {idx_list}):"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{grid_b64}"}},
                {"type": "text",      "text": f"Which cells ({idx_list}) contain the target symbol? Reply ONLY with numbers or NONE."},
            ]},
        ],
        "temperature": 0.0,
        "max_tokens":  32,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    for attempt in range(retries + 1):
        try:
            r = requests.post(BASE_URL, headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"].strip().upper()
            if "NONE" in raw and not any(ch.isdigit() for ch in raw):
                return set()
            confirmed = set()
            for tok in raw.replace(",", " ").split():
                digits = "".join(c for c in tok if c.isdigit())
                if digits:
                    idx = int(digits)
                    if 1 <= idx <= n:
                        confirmed.add(idx)
            return confirmed
        except Exception as e:
            print(f"  (LLM batch error attempt {attempt+1}: {e})")
            if attempt < retries:
                time.sleep(2)
    return None


# ==============================
# IconCounter
# ==============================

class IconCounter:
    def __init__(self, icon_path: str, image_source, max_dimension: int = 8000):
        self.icon_original = cv2.imread(icon_path)
        self.image_original = cv2.imread(image_source) if isinstance(image_source, str) else image_source

        if self.icon_original is None:
            raise ValueError(f"Could not load icon: {icon_path}")
        if self.image_original is None:
            raise ValueError("Could not load floor plan image.")

        self.image_height, self.image_width = self.image_original.shape[:2]
        self.icon_height,  self.icon_width  = self.icon_original.shape[:2]

        max_dim = max(self.image_width, self.image_height)
        self.scale_factor = 1.0

        if max_dim > max_dimension:
            self.scale_factor = max_dimension / max_dim
            print(f"Image {self.image_width}x{self.image_height} → downsampling {self.scale_factor:.2f}x")
            self.image = cv2.resize(self.image_original, None,
                                    fx=self.scale_factor, fy=self.scale_factor,
                                    interpolation=cv2.INTER_AREA)
            self.icon  = cv2.resize(self.icon_original, None,
                                    fx=self.scale_factor, fy=self.scale_factor,
                                    interpolation=cv2.INTER_AREA)
        else:
            self.image = self.image_original
            self.icon  = self.icon_original

        print(f"Processing at: {self.image.shape[1]}x{self.image.shape[0]}")
        print(f"Icon size:     {self.icon.shape[1]}x{self.icon.shape[0]}")

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        h, w = image.shape[:2]
        M   = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw = int(h * sin + w * cos)
        nh = int(h * cos + w * sin)
        M[0, 2] += nw / 2 - w // 2
        M[1, 2] += nh / 2 - h // 2
        return cv2.warpAffine(image, M, (nw, nh),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    def _nms_advanced(self, matches: List[dict], overlap_thresh: float = 0.4) -> List[dict]:
        if not matches:
            return []
        matches = sorted(matches, key=lambda x: x['confidence'], reverse=True)
        filtered = []
        for match in matches:
            x1, y1 = match['x'], match['y']
            w, h   = match['w'], match['h']
            x2, y2 = x1 + w, y1 + h
            cx, cy = x1 + w / 2, y1 + h / 2
            area1  = w * h
            keep   = True
            for kept in filtered:
                kx1, ky1 = kept['x'], kept['y']
                kw,  kh   = kept['w'], kept['h']
                kx2, ky2  = kx1 + kw, ky1 + kh
                iw_  = max(0, min(x2, kx2) - max(x1, kx1))
                ih_  = max(0, min(y2, ky2) - max(y1, ky1))
                union = area1 + kw * kh - iw_ * ih_
                iou   = (iw_ * ih_) / union if union > 0 else 0
                dist  = math.sqrt((cx - (kx1 + kw/2))**2 + (cy - (ky1 + kh/2))**2)
                if iou > overlap_thresh or dist < min(w, h, kw, kh) * 0.4:
                    keep = False
                    break
            if keep:
                filtered.append(match)
        return filtered

    def count_icons_robust(self,
                           angles=None, threshold=0.6,
                           show_matches=True, use_color=False,
                           n_workers=None, image_to_process=None,
                           log_fn=None):
        def log(msg):
            if log_fn: log_fn(msg)
            print(msg)

        if angles is None:
            angles = [0, 45, 90, 135, 180, 225, 270, 315]
        if n_workers is None:
            n_workers = multiprocessing.cpu_count()

        max_dim    = max(self.image.shape[:2])
        use_tiling = max_dim > 6000
        worker_fn  = _match_angle_tiled if use_tiling else _match_angle

        target_img = image_to_process if image_to_process is not None else self.image
        img_match  = target_img if use_color else cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

        rotated = {}
        for a in angles:
            ri = self.rotate_image(self.icon, a)
            rotated[a] = ri if use_color else cv2.cvtColor(ri, cv2.COLOR_BGR2GRAY)

        tasks = [(a, rotated[a], img_match, threshold) for a in angles]

        log(f"  Angle scan: {len(tasks)} tasks via ThreadPool ({n_workers} workers)...")
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(worker_fn, tasks))

        all_matches = [m for r in results for m in r]
        log(f"  Raw hits: {len(all_matches)}")
        filtered = self._nms_advanced(all_matches, overlap_thresh=0.2)
        log(f"  After NMS: {len(filtered)}")

        for m in filtered:
            m['x'] = int(m['x'] / self.scale_factor)
            m['y'] = int(m['y'] / self.scale_factor)
            m['w'] = int(m['w'] / self.scale_factor)
            m['h'] = int(m['h'] / self.scale_factor)

        result_image = self.image_original.copy()
        if show_matches:
            for m in filtered:
                x, y, w, h = m['x'], m['y'], m['w'], m['h']
                cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 0, 255), 3)
                cv2.putText(result_image, f"{m['confidence']:.2f}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return len(filtered), result_image, filtered


# ==============================
# Three-tier verification
# ==============================

def verify_all_candidates(
        symbol_bgr:   np.ndarray,
        symbol_pil:   Image.Image,
        floor_pil:    Image.Image,
        candidates:   List[dict],
        candidate_boxes: List[tuple],
        log_fn=None) -> List[tuple]:
    """
    Tier 1 — Template confidence  : instant, no image analysis
    Tier 2 — CV verification      : ~1–5ms per crop (SSIM + ORB + Hist)
    Tier 3 — LLM batch            : ~10s per batch, only for genuinely ambiguous
    """
    def log(msg):
        if log_fn: log_fn(msg)
        print(msg)

    total = len(candidates)
    log(f"\n🔍 Verifying {total} candidates...")
    log(f"   Tier 1 conf   : auto-confirm ≥{HIGH_CONF_AUTO} | auto-reject <{LOW_CONF_AUTO}")
    log(f"   Tier 2 CV     : confirm ≥{CV_CONFIRM_SCORE} | reject <{CV_REJECT_SCORE}")
    log(f"   Tier 3 LLM    : batch size {LLM_BATCH_SIZE} (last resort only)")

    cv_verifier = CVVerifier(symbol_bgr)
    floor_np    = np.array(floor_pil)  # RGB numpy for CV crops

    confirmed_boxes = []
    llm_queue       = []   # (global_idx, candidate, box, crop_pil, cv_scores)

    t0_cv = time.time()

    for i, (c, box) in enumerate(zip(candidates, candidate_boxes)):
        conf = c['confidence']

        # ── Tier 1: template match confidence ──────────────────────────────
        if conf >= HIGH_CONF_AUTO:
            log(f"  #{i+1:>3}  ✅ T1-AUTO-CONFIRM  conf={conf:.3f}")
            confirmed_boxes.append(box)
            continue
        if conf < LOW_CONF_AUTO:
            log(f"  #{i+1:>3}  ❌ T1-AUTO-REJECT   conf={conf:.3f}")
            continue

        # ── Tier 2: CV verification ─────────────────────────────────────────
        x1, y1, x2, y2 = box
        crop_np  = cv2.cvtColor(floor_np[y1:y2, x1:x2], cv2.COLOR_RGB2BGR)
        cv_scores = cv_verifier.score(crop_np)
        cv_score  = cv_scores['combined']

        if cv_score >= CV_CONFIRM_SCORE:
            log(f"  #{i+1:>3}  ✅ T2-CV-CONFIRM    conf={conf:.3f}  cv={cv_score:.3f}  "
                f"(ssim={cv_scores['ssim']:.2f} orb={cv_scores['orb']:.2f} hist={cv_scores['hist']:.2f})")
            confirmed_boxes.append(box)
        elif cv_score < CV_REJECT_SCORE:
            log(f"  #{i+1:>3}  ❌ T2-CV-REJECT     conf={conf:.3f}  cv={cv_score:.3f}")
        else:
            log(f"  #{i+1:>3}  ❓ T2-CV-AMBIGUOUS  conf={conf:.3f}  cv={cv_score:.3f}  → queued for LLM")
            crop_pil = floor_pil.crop(box)
            llm_queue.append((i, c, box, crop_pil, cv_scores))

    cv_elapsed = time.time() - t0_cv
    log(f"\n  ⏱  CV tier done in {cv_elapsed:.2f}s for {total} candidates")
    log(f"  Confirmed so far: {len(confirmed_boxes)} | LLM queue: {len(llm_queue)}")

    # ── Tier 3: LLM batch (only truly ambiguous cases) ──────────────────────
    if llm_queue:
        log(f"\n  🤖 LLM verification for {len(llm_queue)} ambiguous candidates "
            f"({math.ceil(len(llm_queue)/LLM_BATCH_SIZE)} batch call(s))...")

        batches = [llm_queue[i:i+LLM_BATCH_SIZE] for i in range(0, len(llm_queue), LLM_BATCH_SIZE)]
        t0_llm = time.time()

        for bi, batch in enumerate(batches):
            crop_pils = [item[3] for item in batch]
            result_set = verify_batch_llm(symbol_pil, crop_pils)

            for local_idx, (global_idx, c, box, _, cv_s) in enumerate(batch, start=1):
                if result_set is None:
                    log(f"  #{global_idx+1:>3}  ⚠️  T3-LLM-ERROR  (skipped)")
                elif local_idx in result_set:
                    log(f"  #{global_idx+1:>3}  ✅ T3-LLM-CONFIRM  conf={c['confidence']:.3f}  cv={cv_s['combined']:.3f}")
                    confirmed_boxes.append(box)
                else:
                    log(f"  #{global_idx+1:>3}  ❌ T3-LLM-REJECT   conf={c['confidence']:.3f}  cv={cv_s['combined']:.3f}")

            log(f"  Batch {bi+1}/{len(batches)} done ({time.time()-t0_llm:.1f}s elapsed)")

    # ── Summary ─────────────────────────────────────────────────────────────
    t1_auto = sum(1 for c in candidates if c['confidence'] >= HIGH_CONF_AUTO)
    t1_rej  = sum(1 for c in candidates if c['confidence'] < LOW_CONF_AUTO)
    t2_cnt  = total - t1_auto - t1_rej
    t3_cnt  = len(llm_queue)

    log(f"\n  ══ Verification summary ══")
    log(f"  Tier 1 (instant) : {t1_auto} confirmed, {t1_rej} rejected")
    log(f"  Tier 2 (CV)      : {t2_cnt - t3_cnt} resolved without LLM")
    log(f"  Tier 3 (LLM)     : {t3_cnt} sent to LLM")
    log(f"  TOTAL confirmed  : {len(confirmed_boxes)} / {total}")
    return confirmed_boxes


# ==============================
# MAIN
# ==============================

def count_symbol(symbol_path: str, floor_plan_path: str,
                 output_path: str = "output_detected.png", log_fn=None):
    start = time.time()

    def log(msg):
        if log_fn: log_fn(msg)
        print(msg)

    log(f"\n Processing: {Path(floor_plan_path).name}")
    log(f" Symbol:     {Path(symbol_path).name}")

    symbol_pil = Image.open(symbol_path).convert("RGB")
    symbol_bgr = cv2.imread(symbol_path)
    floor_w, floor_h = Image.open(floor_plan_path).size

    # 1. Color filter
    log("\n[Step 1] Color filtering...")
    icon_colors = extract_icon_colors(symbol_bgr)
    log(f"  {len(icon_colors)} color components detected.")
    floor_cv     = cv2.imread(floor_plan_path)
    filtered_cv  = filter_by_color(floor_cv, icon_colors, tolerance=30)

    # 2. Two-scan template matching
    log("\n[Step 2] Template matching...")
    counter = IconCounter(symbol_path, filtered_cv, max_dimension=8000)

    log("  Scan 1: color-filtered image")
    _, _, matches_filt = counter.count_icons_robust(
        angles=[0, 45, 90, 135, 180, 225, 270, 315],
        threshold=0.6, show_matches=False, use_color=False,
        n_workers=multiprocessing.cpu_count(), log_fn=log_fn
    )

    log("  Scan 2: original image")
    original_cv = cv2.imread(floor_plan_path)
    scaled_orig = cv2.resize(original_cv, None,
                             fx=counter.scale_factor, fy=counter.scale_factor,
                             interpolation=cv2.INTER_AREA)
    _, _, matches_orig = counter.count_icons_robust(
        angles=[0, 45, 90, 135, 180, 225, 270, 315],
        threshold=0.6, show_matches=False, use_color=False,
        n_workers=multiprocessing.cpu_count(), image_to_process=scaled_orig,
        log_fn=log_fn
    )

    final_matches = counter._nms_advanced(matches_filt + matches_orig, overlap_thresh=0.2)
    log(f"  Final unique candidates: {len(final_matches)}")

    candidates = [{
        'x': m['x'], 'y': m['y'], 'w': m['w'], 'h': m['h'],
        'cx': m['x'] + m['w'] // 2, 'cy': m['y'] + m['h'] // 2,
        'confidence': m['confidence']
    } for m in final_matches]

    if not candidates:
        log("No candidates found.")
        return 0

    # 3. Build crop boxes
    floor_pil_clean = Image.open(floor_plan_path).convert("RGB")
    floor_pil_annot = floor_pil_clean.copy()
    draw = ImageDraw.Draw(floor_pil_annot)
    candidate_boxes = []

    try:
        _font_size  = max(20, int(min(floor_w, floor_h) / 120))
        _label_font = ImageFont.truetype("arial.ttf", _font_size)
    except Exception:
        _label_font = None
        _font_size  = 12

    for idx, c in enumerate(candidates):
        x1 = max(0, c['x'] - CROP_PADDING)
        y1 = max(0, c['y'] - CROP_PADDING)
        x2 = min(floor_w, c['x'] + c['w'] + CROP_PADDING)
        y2 = min(floor_h, c['y'] + c['h'] + CROP_PADDING)
        box = (x1, y1, x2, y2)
        candidate_boxes.append(box)
        draw.rectangle(box, outline="blue", width=2)
        label = str(idx + 1)
        tx, ty = x1 + 3, y1 + 3
        draw.rectangle((tx-2, ty-2, tx + _font_size*len(label)//2+4, ty+_font_size+2), fill="blue")
        draw.text((tx, ty), label, fill="white", font=_label_font)

    # 4. Three-tier verification
    log("\n[Step 3] Three-tier verification...")
    confirmed = verify_all_candidates(
        symbol_bgr, symbol_pil, floor_pil_clean,
        candidates, candidate_boxes, log_fn=log_fn
    )

    # 5. Draw confirmed boxes
    for box in confirmed:
        draw.rectangle(box, outline="red", width=10)
        orig_idx = candidate_boxes.index(box)
        label = str(orig_idx + 1)
        tx, ty = box[0] + 3, box[1] + 3
        draw.rectangle((tx-2, ty-2, tx + _font_size*len(label)//2+4, ty+_font_size+2), fill="red")
        draw.text((tx, ty), label, fill="white", font=_label_font)

    floor_pil_annot.save(output_path)

    elapsed = time.time() - start
    log(f"\n{'='*40}")
    log(f"  Output    : {Path(output_path).name}")
    log(f"  Elapsed   : {elapsed:.1f}s")
    log(f"  Candidates: {len(candidates)}")
    log(f"  Confirmed : {len(confirmed)}")
    log(f"{'='*40}")
    return len(confirmed)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    count_symbol(SYMBOL, FLOOR_PLAN)