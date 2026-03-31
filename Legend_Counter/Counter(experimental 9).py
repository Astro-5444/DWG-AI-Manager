"""
Floor Plan Color Filter
------------------------
The symbol template is black & white (no color info).
The actual symbol instances on the floor plan are COLORED.

Two modes:
  MODE = "manual"  → you specify the color name(s) yourself
  MODE = "sample"  → you click a pixel on the floor plan to sample the color,
                     then the script builds the HSV range from that sample
  MODE = "all"     → keeps ALL colored pixels (red + green + blue) on the floor plan
"""

import cv2
import numpy as np
from pathlib import Path

SYMBOL_PATH     = r"D:\AVIS\Manager\output\small_test\symbols\symbol_4.png"
FLOOR_PLAN_PATH = r"D:\AVIS\Manager\output\small_test\43001-AJI-04-DWG-IC-BL1-210004-000\floor_plan.png"
OUTPUT_PATH     = r"D:\AVIS\Manager\output\small_test\43001-AJI-04-DWG-IC-BL1-210004-000\filtered_floor_plan.png"

# ── Choose your mode ──────────────────────────────────────────────────────────
MODE = "all"          # "manual" | "sample" | "all"

# Used when MODE = "manual" — pick one or more: "red", "green", "blue", "yellow", "orange", "purple"
MANUAL_COLORS = ["red"]

# Used when MODE = "sample" — pixel coordinate (x, y) of a known symbol instance on the floor plan
SAMPLE_PIXEL = (300, 200)   # ← change this to a pixel that sits ON a symbol

# ── HSV filter tolerance (used in all modes) ──────────────────────────────────
HUE_MARGIN  = 15    # ± degrees around detected/chosen hue
SAT_MIN     = 40    # minimum saturation
VAL_MIN     = 40    # minimum value (brightness)
# ─────────────────────────────────────────────────────────────────────────────

# Predefined HSV ranges for named colors
NAMED_HSV = {
    "red":    [((0,   SAT_MIN, VAL_MIN), (10,  255, 255)),
               ((165, SAT_MIN, VAL_MIN), (180, 255, 255))],
    "orange": [((10,  SAT_MIN, VAL_MIN), (25,  255, 255))],
    "yellow": [((25,  SAT_MIN, VAL_MIN), (35,  255, 255))],
    "green":  [((35,  SAT_MIN, VAL_MIN), (85,  255, 255))],
    "cyan":   [((85,  SAT_MIN, VAL_MIN), (100, 255, 255))],
    "blue":   [((100, SAT_MIN, VAL_MIN), (130, 255, 255))],
    "purple": [((130, SAT_MIN, VAL_MIN), (155, 255, 255))],
    "pink":   [((155, SAT_MIN, VAL_MIN), (165, 255, 255))],
}


def load_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        alpha = img[:, :, 3:4] / 255.0
        bgr   = img[:, :, :3].astype(float)
        white = np.ones_like(bgr) * 255
        img   = (bgr * alpha + white * (1 - alpha)).astype(np.uint8)
    return img


def hue_to_name(hue: int) -> str:
    if hue < 10 or hue >= 165: return "red"
    if hue < 25:                return "orange"
    if hue < 35:                return "yellow"
    if hue < 85:                return "green"
    if hue < 100:               return "cyan"
    if hue < 130:               return "blue"
    if hue < 155:               return "purple"
    return "pink"


def analyze_floor_plan_colors(floor_bgr: np.ndarray):
    """Show what chromatic colors exist on the floor plan."""
    hsv   = cv2.cvtColor(floor_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    total = len(h)

    chromatic = (s > SAT_MIN) & (v > VAL_MIN)
    ch = h[chromatic]

    print(f"\n  Floor plan chromatic pixels: {chromatic.sum():,} / {total:,}  ({chromatic.sum()/total*100:.2f}%)")
    if len(ch) == 0:
        print("  [!] No chromatic pixels found on floor plan either.")
        return

    print(f"\n  Color distribution on floor plan:")
    hist, _ = np.histogram(ch, bins=18, range=(0, 180))
    for i, cnt in enumerate(hist):
        if cnt == 0:
            continue
        bar   = "█" * max(1, int(cnt / max(hist) * 30))
        hname = hue_to_name(i * 10)
        pct   = cnt / total * 100
        print(f"    H {i*10:>3}–{i*10+10:<3}  {cnt:>7,}  {pct:>5.2f}%  {bar}  ({hname})")


def build_mask_from_ranges(floor_hsv: np.ndarray, ranges: list) -> np.ndarray:
    mask = np.zeros(floor_hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(floor_hsv, np.array(lo), np.array(hi))
    return mask


def build_mask_from_sample(floor_hsv: np.ndarray, pixel_xy: tuple) -> np.ndarray:
    x, y    = pixel_xy
    h, w    = floor_hsv.shape[:2]
    x, y    = min(x, w - 1), min(y, h - 1)
    hsv_px  = floor_hsv[y, x]
    hue     = int(hsv_px[0])
    sat     = int(hsv_px[1])
    val     = int(hsv_px[2])

    print(f"\n  Sampled pixel ({x},{y}) → HSV({hue}, {sat}, {val})  [{hue_to_name(hue)}]")

    if sat < SAT_MIN or val < VAL_MIN:
        print(f"  [!] Sampled pixel has low saturation or value — it may not be a colored symbol.")
        print(f"      Try a different coordinate that sits on the colored part of a symbol.")

    lo_hue = max(0,   hue - HUE_MARGIN)
    hi_hue = min(180, hue + HUE_MARGIN)

    # Handle red hue wrap-around
    if lo_hue < 5 or hi_hue > 175:
        ranges = [
            ((0,      SAT_MIN, VAL_MIN), (10,  255, 255)),
            ((165,    SAT_MIN, VAL_MIN), (180, 255, 255)),
        ]
    else:
        ranges = [((lo_hue, SAT_MIN, VAL_MIN), (hi_hue, 255, 255))]

    return build_mask_from_ranges(floor_hsv, ranges)


def main():
    print("\n── Loading images ──────────────────────────────────────────")
    floor_plan = load_bgr(FLOOR_PLAN_PATH)
    print(f"  Floor plan : {floor_plan.shape[1]}×{floor_plan.shape[0]}")
    print(f"  Symbol     : (black & white — no color to extract from template)")

    floor_hsv = cv2.cvtColor(floor_plan, cv2.COLOR_BGR2HSV)

    # Always show what colors the floor plan has
    analyze_floor_plan_colors(floor_plan)

    # ── Build mask based on chosen mode ──────────────────────────────────────
    print(f"\n── Mode: {MODE} ─────────────────────────────────────────────")

    if MODE == "all":
        # Keep every chromatic pixel on the floor plan
        ranges = []
        for color_ranges in NAMED_HSV.values():
            ranges.extend(color_ranges)
        combined_mask = build_mask_from_ranges(floor_hsv, ranges)
        print("  Keeping ALL chromatic pixels (red + green + blue + all colors).")

    elif MODE == "manual":
        combined_mask = np.zeros(floor_plan.shape[:2], dtype=np.uint8)
        for color in MANUAL_COLORS:
            if color not in NAMED_HSV:
                print(f"  [!] Unknown color '{color}'. Available: {list(NAMED_HSV.keys())}")
                continue
            m = build_mask_from_ranges(floor_hsv, NAMED_HSV[color])
            kept = int(m.sum())
            print(f"  {color:<10}  kept {kept:>8,} pixels")
            combined_mask |= m

    elif MODE == "sample":
        combined_mask = build_mask_from_sample(floor_hsv, SAMPLE_PIXEL)

    else:
        print(f"  [!] Unknown MODE '{MODE}'. Use 'all', 'manual', or 'sample'.")
        return

    # ── Stats ─────────────────────────────────────────────────────────────────
    total_kept = int(combined_mask.sum())
    total_fp   = floor_plan.shape[0] * floor_plan.shape[1]
    print(f"\n  Total kept : {total_kept:,} / {total_fp:,} pixels  ({total_kept/total_fp*100:.2f}%)")

    # ── Apply mask ────────────────────────────────────────────────────────────
    result       = np.full_like(floor_plan, 255)
    result_black = np.zeros_like(floor_plan)
    result[combined_mask > 0]       = floor_plan[combined_mask > 0]
    result_black[combined_mask > 0] = floor_plan[combined_mask > 0]

    # ── Save ──────────────────────────────────────────────────────────────────
    out = Path(OUTPUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), result)
    cv2.imwrite(str(out.with_stem(out.stem + "_black_bg")), result_black)
    cv2.imwrite(str(out.with_stem(out.stem + "_mask")),     combined_mask)

    print(f"\n── Saved ───────────────────────────────────────────────────")
    print(f"  White bg : {out}")
    print(f"  Black bg : {out.with_stem(out.stem + '_black_bg')}")
    print(f"  Mask     : {out.with_stem(out.stem + '_mask')}")
    print()


if __name__ == "__main__":
    main()