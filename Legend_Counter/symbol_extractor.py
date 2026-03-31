import cv2
import numpy as np
from collections import defaultdict
import os

DEFAULT_MIN_LINE_LENGTH = 300


# ==============================================================================
# VIEWER
# ==============================================================================

class ZoomPanViewer:
    def __init__(self, window_name, original_image, annotated_image):
        self.window_name = window_name
        self.original_image = original_image
        self.annotated_image = annotated_image
        self.zoom = 0.8
        self.offset = np.array([0, 0], dtype=np.float32)
        self.dragging = False
        self.last_mouse = np.array([0, 0], dtype=np.float32)
        self.show_help = True
        self.show_annotations = True

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

    def on_mouse(self, event, x, y, flags, param):
        mouse = np.array([x, y], dtype=np.float32)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.last_mouse = mouse
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                self.offset -= (mouse - self.last_mouse) / self.zoom
                self.last_mouse = mouse
        elif event == cv2.EVENT_MOUSEWHEEL:
            factor = 1.1 if flags > 0 else 0.9
            img_pos = self.offset + mouse / self.zoom
            self.zoom = max(0.01, min(self.zoom * factor, 100.0))
            self.offset = img_pos - mouse / self.zoom

    def draw_overlays(self, display):
        if not self.show_help:
            return
        overlay = display.copy()
        cv2.rectangle(overlay, (10, 10), (420, 215), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
        mode_str = "HIGHLIGHT ON" if self.show_annotations else "ORIGINAL"
        lines = [
            f"Zoom: {self.zoom:.2f}x  |  {mode_str}",
            "--- Controls ---",
            "Scroll: Zoom | Drag: Pan",
            "T: Toggle 2nd Table Highlight",
            "R: Reset | H: Help | Q/Esc: Quit",
            "--- Legend ---",
            "Green lines = All Horizontal lines",
            "Blue lines = All Vertical lines",
            "Cyan lines = 2nd Biggest Table",
            "Cyan border = Table Bounding Box",
        ]
        for i, text in enumerate(lines):
            cv2.putText(display, text, (20, 35 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    def run(self):
        while True:
            rect = cv2.getWindowImageRect(self.window_name)
            ww, wh = (rect[2], rect[3]) if rect[2] > 0 else (1280, 720)
            M = np.float32([
                [self.zoom, 0, -self.offset[0] * self.zoom],
                [0, self.zoom, -self.offset[1] * self.zoom]
            ])
            base = self.original_image if not self.show_annotations else self.annotated_image
            display = cv2.warpAffine(base, M, (ww, wh),
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(30, 30, 30))
            self.draw_overlays(display)
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('r'):
                self.zoom = 0.8
                self.offset = np.array([0, 0], dtype=np.float32)
            elif key == ord('h'):
                self.show_help = not self.show_help
            elif key == ord('t'):
                self.show_annotations = not self.show_annotations
        cv2.destroyAllWindows()


# ==============================================================================
# LINE DETECTION
# ==============================================================================

def get_line_mask(thresh, direction, min_length):
    if direction == 'h':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_length, 1))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_length))
    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)


def get_line_rects(mask, min_area=50):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > min_area]


def rects_intersect(r1, r2, padding=4):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    return not (
        x1 + w1 + padding < x2 or
        x2 + w2 + padding < x1 or
        y1 + h1 + padding < y2 or
        y2 + h2 + padding < y1
    )


# ==============================================================================
# TABLE BUILDING via UNION-FIND
# ==============================================================================

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def build_tables(h_rects, v_rects):
    nh = len(h_rects)
    nv = len(v_rects)
    uf = UnionFind(nh + nv)

    intersection_h = set()
    intersection_v = set()

    for i, hr in enumerate(h_rects):
        for j, vr in enumerate(v_rects):
            if rects_intersect(hr, vr):
                uf.union(i, nh + j)
                intersection_h.add(i)
                intersection_v.add(j)

    components = defaultdict(lambda: {"h": [], "v": []})
    for i in range(nh):
        if i in intersection_h:
            components[uf.find(i)]["h"].append(i)
    for j in range(nv):
        if j in intersection_v:
            components[uf.find(nh + j)]["v"].append(j)

    tables = [c for c in components.values() if c["h"] and c["v"]]
    return tables, intersection_h, intersection_v


def table_bounding_box(table, h_rects, v_rects):
    all_rects = [h_rects[i] for i in table["h"]] + [v_rects[j] for j in table["v"]]
    xs  = [r[0]          for r in all_rects]
    ys  = [r[1]          for r in all_rects]
    x2s = [r[0] + r[2]   for r in all_rects]
    y2s = [r[1] + r[3]   for r in all_rects]
    return min(xs), min(ys), max(x2s), max(y2s)


def table_score(table, h_rects, v_rects):
    x1, y1, x2, y2 = table_bounding_box(table, h_rects, v_rects)
    area = (x2 - x1) * (y2 - y1)
    n_lines = len(table["h"]) + len(table["v"])
    return area * n_lines


def draw_table_highlight(canvas, table, h_rects, v_rects, line_color, box_color, label):
    for i in table["h"]:
        x, y, w, h = h_rects[i]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), line_color, 2)
    for j in table["v"]:
        x, y, w, h = v_rects[j]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), line_color, 2)

    x1, y1, x2, y2 = table_bounding_box(table, h_rects, v_rects)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
    cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0, canvas)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 4)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    lx, ly = x1, max(y1 - 12, th + 4)
    cv2.rectangle(canvas, (lx - 4, ly - th - 6), (lx + tw + 4, ly + 4), (0, 0, 0), -1)
    cv2.putText(canvas, label, (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2, cv2.LINE_AA)


def extract_left_column_symbols(table, h_rects, v_rects, original_img):
    x1, y1, x2, y2 = table_bounding_box(table, h_rects, v_rects)

    table_h_lines = [h_rects[i] for i in table["h"]]
    table_v_lines = [v_rects[j] for j in table["v"]]

    v_centers = sorted(set(r[0] + r[2] // 2 for r in table_v_lines))
    deduped_v = [v_centers[0]]
    for vc in v_centers[1:]:
        if vc - deduped_v[-1] > 15:
            deduped_v.append(vc)

    BORDER_TOLERANCE = 20
    inner_v = [vc for vc in deduped_v
               if vc > x1 + BORDER_TOLERANCE and vc < x2 - BORDER_TOLERANCE]
    col_boundaries = [x1] + inner_v + [x2]

    h_centers = sorted(set(r[1] + r[3] // 2 for r in table_h_lines))
    deduped_h = [h_centers[0]]
    for hc in h_centers[1:]:
        if hc - deduped_h[-1] > 10:
            deduped_h.append(hc)
    row_boundaries = [y1] + deduped_h + [y2]

    col_x1 = col_boundaries[0]
    col_x2 = col_boundaries[1]

    cropped_symbols = []
    margin = 5
    for i in range(len(row_boundaries) - 1):
        y_top = row_boundaries[i]
        y_bottom = row_boundaries[i + 1]
        if y_bottom - y_top < 20:
            continue
        crop_x1 = int(max(0, col_x1 + margin))
        crop_y1 = int(max(0, y_top + margin))
        crop_x2 = int(min(original_img.shape[1], col_x2 - margin))
        crop_y2 = int(min(original_img.shape[0], y_bottom - margin))
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            continue
        symbol_img = original_img[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        cropped_symbols.append((symbol_img, i))

    return cropped_symbols


def crop_legend_table(table, h_rects, v_rects, original_img):
    """
    Crop the full bounding box of a table from the original image.

    Args:
        table       (dict): Table dict with 'h' and 'v' line index lists.
        h_rects     (list): Horizontal line bounding rects.
        v_rects     (list): Vertical line bounding rects.
        original_img (np.ndarray): Source BGR image.

    Returns:
        np.ndarray: Cropped image of the full legend table.
    """
    x1, y1, x2, y2 = table_bounding_box(table, h_rects, v_rects)
    crop_x1 = int(max(0, x1))
    crop_y1 = int(max(0, y1))
    crop_x2 = int(min(original_img.shape[1], x2))
    crop_y2 = int(min(original_img.shape[0], y2))
    return original_img[crop_y1:crop_y2, crop_x1:crop_x2].copy()


# ==============================================================================
# PUBLIC API
# ==============================================================================

def detect_lines(img, min_line_length=DEFAULT_MIN_LINE_LENGTH):
    """
    Run line detection on a BGR image.

    Returns:
        h_rects (list): Bounding rects of detected horizontal lines.
        v_rects (list): Bounding rects of detected vertical lines.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    edges = cv2.Canny(enhanced, 10, 40)
    thresh = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    h_mask = get_line_mask(thresh, 'h', min_line_length)
    v_mask = get_line_mask(thresh, 'v', min_line_length)

    return get_line_rects(h_mask), get_line_rects(v_mask)


def detect_tables(img, min_line_length=DEFAULT_MIN_LINE_LENGTH):
    """
    Detect all tables in a BGR image, sorted largest-first by score.

    Returns:
        sorted_tables (list[dict]): Each dict has keys 'h' and 'v' (line indices).
        h_rects (list): Horizontal line bounding rects.
        v_rects (list): Vertical line bounding rects.
    """
    h_rects, v_rects = detect_lines(img, min_line_length)
    tables, _, _ = build_tables(h_rects, v_rects)
    sorted_tables = sorted(tables, key=lambda t: table_score(t, h_rects, v_rects), reverse=True)
    return sorted_tables, h_rects, v_rects


def get_table_bbox(table, h_rects, v_rects):
    """
    Return (x1, y1, x2, y2) bounding box for a table dict.
    Convenience wrapper around table_bounding_box().
    """
    return table_bounding_box(table, h_rects, v_rects)


def process_image(image_path, save_folder):
    """
    Detect the 2nd biggest table in an image, save left-column symbol crops,
    and save a crop of the full legend table — all in save_folder.

    Args:
        image_path  (str): Path to the input image.
        save_folder (str): Folder where cropped symbol images will be saved.

    Returns:
        list of saved file paths (symbols + legend table crop).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: '{image_path}'")

    sorted_tables, h_rects, v_rects = detect_tables(img)

    if len(sorted_tables) < 2:
        print("Warning: fewer than 2 tables found.")
        return []

    target = sorted_tables[1]
    symbols = extract_left_column_symbols(target, h_rects, v_rects, img)

    os.makedirs(save_folder, exist_ok=True)
    saved_paths = []

    # Save individual symbol crops
    for sym_img, row_idx in symbols:
        filepath = os.path.join(save_folder, f"symbol_{row_idx+1}.png")
        cv2.imwrite(filepath, sym_img)
        saved_paths.append(filepath)
        print(f"Saved: {filepath}")

    # Save the full legend table crop in the same folder
    legend_crop = crop_legend_table(target, h_rects, v_rects, img)
    legend_path = os.path.join(save_folder, "legend_table.png")
    cv2.imwrite(legend_path, legend_crop)
    saved_paths.append(legend_path)
    print(f"Saved: {legend_path}")
    
    """
    # If successful, run the naming script automatically
    print("\n" + "="*50)
    print("Extraction successful. Starting symbol naming...")
    print("="*50 + "\n")
    try:
        import symbol_namer
        symbol_namer.main(save_folder)
    except ImportError:
        print("Error: symbol_namer.py not found in the same directory.")
    except Exception as e:
        print(f"Error running symbol_namer: {e}")
    """

    return saved_paths


# ==============================================================================
# STANDALONE ENTRY POINT
# ==============================================================================

def main():
    process_image(r"C:\Users\kabdu\OneDrive\Desktop\AVIS\Manager\output\New folder (3)\43001-AJI-04-DWG-IC-L01-210021-000\Information_Box.png", "symbol")


if __name__ == "__main__":
    main()


# ==============================================================================
# EXAMPLE USAGE FROM ANOTHER SCRIPT
# ==============================================================================
#
#   from Legend_Extractor import process_image
#
#   saved = process_image("Information_Box.png", "output/symbols")
#   print(saved)  # ['output/symbols/symbol_1.png', ..., 'output/symbols/legend_table.png']
#