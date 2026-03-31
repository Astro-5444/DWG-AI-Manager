"""
symbol_viewer_pyqt5.py — Optimized High-Resolution Image Viewer (PyQt5)
==========================================================================
Usage (standalone):
    python symbol_viewer_pyqt5.py <image_path> <symbol_output_folder>

Usage (from another script):
    from symbol_viewer_pyqt5 import launch_viewer
    launch_viewer("/path/to/image.png", "/path/to/symbols/")
"""

import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
    QInputDialog, QVBoxLayout, QWidget, QLabel, QPushButton,
    QHBoxLayout, QListWidget, QListWidgetItem, QMessageBox,
    QFrame, QSplitter,
)
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QImage, QFont, QCursor
from PyQt5.QtCore import Qt, QRect, QRectF, QPoint, QTimer, QPointF
from PIL import Image
import numpy as np

Image.MAX_IMAGE_PIXELS = 500_000_000


# ─────────────────────────────────────────────────────────────────────────────
# MAIN VIEWER
# ─────────────────────────────────────────────────────────────────────────────

class SymbolViewer(QMainWindow):
    ZOOM_STEP   = 1.25
    ZOOM_MIN    = 0.002
    ZOOM_MAX    = 50.0
    SEL_COLOR   = QColor(255, 51,  51)   # #FF3333  – live drag rect
    SEL_WIDTH   = 2
    PENDING_CLR = QColor(255, 153,  0)   # #FF9900  – unsaved
    SAVED_CLR   = QColor( 51, 238, 102)  # #33EE66  – saved
    HOVER_CLR   = QColor(255, 255, 255)  # #FFFFFF  – hovered

    def __init__(self, image_path: str, output_dir: str):
        super().__init__()
        self.image_path = image_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.setWindowTitle(f"Symbol Viewer · {os.path.basename(image_path)}")
        self.setGeometry(100, 100, 1400, 900)

        # ── image state ───────────────────────────────────────────────────────
        self.pil_image    = None
        self.img_w        = 0
        self.img_h        = 0
        self.image_pixmap = None
        self.image_item   = None

        # ── selection state ───────────────────────────────────────────────────
        self.selections = []   # list of dicts: img_bbox, name, saved
        self.hover_sel  = None

        self._build_ui()
        self._load_image()

    # ─────────────────────────────────────────────────────────────────────────
    # UI
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.setStyleSheet("background:#111116; color:#c8c8e0;")

        # ── scene / view ──────────────────────────────────────────────────────
        self.scene = QGraphicsScene(self)
        self.view  = GraphicsView(self)
        self.view.setScene(self.scene)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.view.setDragMode(QGraphicsView.NoDrag)           # we handle panning
        self.view.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.view.setResizeAnchor(QGraphicsView.NoAnchor)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setFrameShape(QFrame.NoFrame)

        # ── central layout ────────────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left: toolbar + view + statusbar
        left = QWidget()
        left.setStyleSheet("background:#0c0c10;")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        left_layout.addWidget(self._build_toolbar())
        left_layout.addWidget(QFrame(frameShape=QFrame.HLine,
                                     styleSheet="background:#2233aa; max-height:2px; border:none;"))
        left_layout.addWidget(self.view, stretch=1)

        status_frame = QWidget()
        status_frame.setStyleSheet("background:#0a0a0e;")
        status_frame.setFixedHeight(26)
        sl = QHBoxLayout(status_frame)
        sl.setContentsMargins(12, 0, 12, 0)
        self.status_lbl = QLabel("Loading…")
        self.status_lbl.setStyleSheet("color:#5566aa; font:8pt 'Courier New';")
        self.zoom_lbl = QLabel("")
        self.zoom_lbl.setStyleSheet("color:#334477; font:8pt 'Courier New';")
        sl.addWidget(self.status_lbl)
        sl.addStretch()
        sl.addWidget(self.zoom_lbl)
        left_layout.addWidget(status_frame)

        # Right: sidebar
        sidebar = self._build_sidebar()

        splitter.addWidget(left)
        splitter.addWidget(sidebar)
        splitter.setSizes([1100, 300])

    def _build_toolbar(self):
        bar = QWidget()
        bar.setFixedHeight(50)
        bar.setStyleSheet("background:#0c0c10;")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(18, 0, 8, 0)
        layout.setSpacing(4)

        title = QLabel("⬡ SYMBOL VIEWER")
        title.setStyleSheet("color:#4466ff; font:bold 12pt 'Courier New';")
        layout.addWidget(title)

        sep = QFrame(frameShape=QFrame.VLine)
        sep.setStyleSheet("color:#222230;")
        layout.addWidget(sep)

        btn_style = """
            QPushButton {
                background:#1e1e26; color:#c8c8e0;
                border:none; padding:8px 14px;
                font:bold 9pt 'Courier New';
            }
            QPushButton:hover { background:#2a2a38; }
            QPushButton:pressed { background:#33334a; }
        """
        for label, slot in [("⊕  In",    self._zoom_in),
                             ("⊖  Out",   self._zoom_out),
                             ("⟲  Reset", self._reset_view),
                             ("⊡  Fit",   self._fit_to_window)]:
            b = QPushButton(label)
            b.setStyleSheet(btn_style)
            b.clicked.connect(slot)
            layout.addWidget(b)

        layout.addStretch()

        for label, slot in [("🗑  Clear",    self._clear_all),
                             ("💾  Save All", self._save_all)]:
            b = QPushButton(label)
            b.setStyleSheet(btn_style)
            b.clicked.connect(slot)
            layout.addWidget(b)

        return bar

    def _build_sidebar(self):
        sidebar = QWidget()
        sidebar.setMinimumWidth(250)
        sidebar.setMaximumWidth(300)
        sidebar.setStyleSheet("background:#13131a;")
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        hdr = QLabel("SELECTIONS")
        hdr.setStyleSheet("color:#4455cc; font:bold 9pt 'Courier New'; padding:12px;")
        hdr.setAlignment(Qt.AlignCenter)
        layout.addWidget(hdr)

        self.sel_list = QListWidget()
        self.sel_list.setStyleSheet("""
            QListWidget {
                background:#0e0e16; color:#aaaacc;
                font:9pt 'Courier New';
                border:none; outline:0;
            }
            QListWidget::item { padding:4px 8px; }
            QListWidget::item:selected { background:#223399; color:#fff; }
            QListWidget::item:hover { background:#1a1a2a; }
        """)
        layout.addWidget(self.sel_list, stretch=1)

        # FIX 7: use currentRowChanged — fires even when re-selecting same row
        self.sel_list.currentRowChanged.connect(self._on_list_row_changed)

        layout.addWidget(QFrame(frameShape=QFrame.HLine,
                                styleSheet="background:#1e1e2a; max-height:1px; border:none; margin:6px 8px;"))

        action_style = """
            QPushButton {
                background:#1a1a24; color:#c8c8e0;
                border:none; padding:7px 8px;
                font:9pt 'Courier New'; text-align:left;
                margin:2px 8px;
            }
            QPushButton:hover { background:#282838; }
        """
        for label, slot in [("✏  Rename Selected",  self._rename_selected),
                             ("💾  Save Selected",    self._save_selected),
                             ("🗑  Delete Selected",  self._delete_selected)]:
            b = QPushButton(label)
            b.setStyleSheet(action_style)
            b.clicked.connect(slot)
            layout.addWidget(b)

        layout.addWidget(QFrame(frameShape=QFrame.HLine,
                                styleSheet="background:#1e1e2a; max-height:1px; border:none; margin:8px;"))

        hdr2 = QLabel("SHORTCUTS")
        hdr2.setStyleSheet("color:#333355; font:bold 8pt 'Courier New'; padding-left:12px;")
        layout.addWidget(hdr2)

        for h in ["Scroll → zoom", "Middle drag → pan", "Left drag → select",
                  "Right click → delete", "Ctrl+S → save all", "Ctrl+Z → undo", "F → fit"]:
            lbl = QLabel(h)
            lbl.setStyleSheet("color:#2d2d4a; font:8pt 'Courier New'; padding-left:12px;")
            layout.addWidget(lbl)

        layout.addStretch()
        return sidebar

    # ─────────────────────────────────────────────────────────────────────────
    # IMAGE LOADING
    # ─────────────────────────────────────────────────────────────────────────

    def _load_image(self):
        self.status_lbl.setText("Opening image…")
        QApplication.processEvents()
        try:
            self.pil_image = Image.open(self.image_path)
            self.pil_image.load()
            self.img_w, self.img_h = self.pil_image.size

            self.image_pixmap = self._pil_to_qpixmap(self.pil_image)

            # FIX 9: give scene an explicit rect so scrollbars stay sane
            self.scene.setSceneRect(0, 0, self.img_w, self.img_h)
            self.image_item = self.scene.addPixmap(self.image_pixmap)
            self.image_item.setZValue(-1)

            self.status_lbl.setText(
                f"{os.path.basename(self.image_path)}  "
                f"[{self.img_w} × {self.img_h} px]   → {self.output_dir}"
            )
        except Exception as ex:
            QMessageBox.critical(self, "Error", f"Could not open image:\n{ex}")
            self.close()
            return

        QTimer.singleShot(120, self._fit_to_window)

    @staticmethod
    def _pil_to_qpixmap(img: Image.Image) -> QPixmap:
        """Convert PIL image to QPixmap, handling RGB / RGBA / L modes."""
        # FIX 5: normalise mode before converting
        mode = img.mode
        if mode == "P":
            img = img.convert("RGBA")
            mode = "RGBA"
        elif mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
            mode = "RGB"

        # FIX 4: ensure C-contiguous array so QImage doesn't read garbage
        arr = np.ascontiguousarray(np.array(img))

        if mode == "RGBA":
            h, w, _ = arr.shape
            q_img = QImage(arr.data, w, h, 4 * w, QImage.Format_RGBA8888)
        elif mode == "RGB":
            h, w, _ = arr.shape
            q_img = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
        else:  # "L"
            h, w = arr.shape
            q_img = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)

        # Copy so the QImage owns its buffer (arr goes out of scope)
        return QPixmap.fromImage(q_img.copy())

    # ─────────────────────────────────────────────────────────────────────────
    # VIEW MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    def _set_zoom(self, zoom: float, anchor_scene: QPointF = None):
        """Apply zoom, keeping anchor_scene fixed under the cursor."""
        zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, zoom))

        if anchor_scene is None:
            anchor_scene = QPointF(self.img_w / 2, self.img_h / 2)

        # FIX 1 & 6: reset transform fully each time, then recompose cleanly
        old_zoom = self._current_zoom()
        scale_factor = zoom / old_zoom if old_zoom else zoom

        self.view.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.view.resetTransform()
        self.view.scale(zoom, zoom)

        # Recentre so anchor_scene stays under mouse
        view_anchor = self.view.mapFromScene(anchor_scene)
        delta = view_anchor - self.view.viewport().rect().center()
        # Shift the view by translating the scene origin
        ox = anchor_scene.x() - delta.x() / zoom
        oy = anchor_scene.y() - delta.y() / zoom
        self.view.resetTransform()
        self.view.scale(zoom, zoom)
        self.view.centerOn(ox, oy)

        self.zoom_lbl.setText(f"zoom  {zoom:.2f}×")
        self.view.viewport().update()

    def _current_zoom(self) -> float:
        return self.view.transform().m11()

    def _fit_to_window(self):
        if self.img_w == 0 or self.img_h == 0:
            return
        vw = self.view.viewport().width()
        vh = self.view.viewport().height()
        zoom = min(vw / self.img_w, vh / self.img_h) * 0.95
        self.view.resetTransform()
        self.view.scale(zoom, zoom)
        self.view.centerOn(self.img_w / 2, self.img_h / 2)
        self.zoom_lbl.setText(f"zoom  {zoom:.2f}×")
        self.view.viewport().update()

    def _zoom_in(self):
        self._set_zoom(self._current_zoom() * self.ZOOM_STEP)

    def _zoom_out(self):
        self._set_zoom(self._current_zoom() / self.ZOOM_STEP)

    def _reset_view(self):
        self.view.resetTransform()
        self.view.centerOn(self.img_w / 2, self.img_h / 2)
        self.zoom_lbl.setText("zoom  1.00×")
        self.view.viewport().update()

    # ─────────────────────────────────────────────────────────────────────────
    # KEYBOARD SHORTCUTS
    # ─────────────────────────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        # FIX 2: check modifiers separately from key
        mods = event.modifiers()
        key  = event.key()

        if mods & Qt.ControlModifier and key == Qt.Key_S:
            self._save_all()
        elif mods & Qt.ControlModifier and key == Qt.Key_Z:
            self._undo_last()
        elif key == Qt.Key_F:
            self._fit_to_window()
        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            self._zoom_in()
        elif key == Qt.Key_Minus:
            self._zoom_out()
        else:
            super().keyPressEvent(event)

    # ─────────────────────────────────────────────────────────────────────────
    # SELECTION MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    def add_selection(self, x0i, y0i, x1i, y1i):
        x0i = max(0, int(x0i)); y0i = max(0, int(y0i))
        x1i = min(self.img_w, int(x1i)); y1i = min(self.img_h, int(y1i))
        if (x1i - x0i) < 4 or (y1i - y0i) < 4:
            return
        sel = {
            "img_bbox": (x0i, y0i, x1i, y1i),
            "name":     f"symbol_{len(self.selections)+1:03d}",
            "saved":    False,
        }
        self.selections.append(sel)
        self._refresh_listbox()

    def _delete_sel(self, sel):
        if sel is self.hover_sel:
            self.hover_sel = None
        self.selections.remove(sel)
        self._refresh_listbox()
        self.view.viewport().update()

    # ─────────────────────────────────────────────────────────────────────────
    # LISTBOX
    # ─────────────────────────────────────────────────────────────────────────

    def _refresh_listbox(self):
        current = self.sel_list.currentRow()
        self.sel_list.clear()
        for sel in self.selections:
            mark = "✓" if sel["saved"] else "·"
            self.sel_list.addItem(QListWidgetItem(f"  {mark}  {sel['name']}"))
        # restore selection
        if 0 <= current < self.sel_list.count():
            self.sel_list.setCurrentRow(current)

    def _on_list_row_changed(self, row):
        if 0 <= row < len(self.selections):
            sel = self.selections[row]
            x0i, y0i, x1i, y1i = sel["img_bbox"]
            self.view.centerOn((x0i + x1i) / 2, (y0i + y1i) / 2)

    # ─────────────────────────────────────────────────────────────────────────
    # ACTIONS
    # ─────────────────────────────────────────────────────────────────────────

    def _rename_selected(self):
        row = self.sel_list.currentRow()
        if row < 0 or row >= len(self.selections):
            QMessageBox.information(self, "Rename", "Select a symbol first."); return
        sel = self.selections[row]
        name, ok = QInputDialog.getText(self, "Rename Symbol", "Symbol name:",
                                        text=sel["name"])
        if ok and name.strip():
            sel["name"]  = name.strip()
            sel["saved"] = False
            self._refresh_listbox()
            self.view.viewport().update()

    def _save_single(self, sel) -> bool:
        x0i, y0i, x1i, y1i = sel["img_bbox"]
        name, ok = QInputDialog.getText(self, "Save Symbol",
                                        "File name (without extension):",
                                        text=sel["name"])
        if not ok or not name.strip():
            return False
        sel["name"] = name.strip()
        crop = self.pil_image.crop((x0i, y0i, x1i, y1i))
        ext  = os.path.splitext(self.image_path)[1] or ".png"
        crop.save(os.path.join(self.output_dir, sel["name"] + ext))
        sel["saved"] = True
        return True

    def _save_selected(self):
        row = self.sel_list.currentRow()
        if row < 0 or row >= len(self.selections):
            QMessageBox.information(self, "Save", "Select a symbol first."); return
        sel = self.selections[row]
        if self._save_single(sel):
            self._refresh_listbox()
            self.view.viewport().update()
            self.status_lbl.setText(f"Saved → {sel['name']}  ({self.output_dir})")

    def _save_all(self):
        if not self.selections:
            QMessageBox.information(self, "Save All", "No selections to save."); return
        saved = 0
        for sel in self.selections:
            if not sel["saved"]:
                if self._save_single(sel):
                    saved += 1
            else:
                x0i, y0i, x1i, y1i = sel["img_bbox"]
                crop = self.pil_image.crop((x0i, y0i, x1i, y1i))
                ext  = os.path.splitext(self.image_path)[1] or ".png"
                crop.save(os.path.join(self.output_dir, sel["name"] + ext))
                saved += 1
        self._refresh_listbox()
        self.view.viewport().update()
        self.status_lbl.setText(f"Saved {saved} symbol(s) → {self.output_dir}")

    def _delete_selected(self):
        row = self.sel_list.currentRow()
        if 0 <= row < len(self.selections):
            self._delete_sel(self.selections[row])

    def _undo_last(self):
        if self.selections:
            self._delete_sel(self.selections[-1])

    def _clear_all(self):
        if not self.selections:
            return
        if QMessageBox.question(self, "Clear All", "Remove all selections?") == QMessageBox.Yes:
            # FIX 10: clear the list properly
            self.selections.clear()
            self.hover_sel = None
            self._refresh_listbox()
            self.view.viewport().update()


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM GRAPHICS VIEW (handles input + drawing)
# ─────────────────────────────────────────────────────────────────────────────

class GraphicsView(QGraphicsView):

    def __init__(self, viewer: SymbolViewer):
        super().__init__()
        self.viewer       = viewer
        self._pan_last    = None
        self._sel_start   = None    # QPointF in scene coords
        self._sel_current = None    # QPointF in scene coords (drag end)
        self.setMouseTracking(True)

    # ── zoom ─────────────────────────────────────────────────────────────────

    def wheelEvent(self, event):
        factor = self.viewer.ZOOM_STEP if event.angleDelta().y() > 0 else 1 / self.viewer.ZOOM_STEP
        anchor = self.mapToScene(event.pos())
        new_zoom = max(self.viewer.ZOOM_MIN,
                       min(self.viewer.ZOOM_MAX,
                           self.viewer._current_zoom() * factor))

        # FIX 1: proper zoom-to-cursor using Qt's built-in anchor mechanism
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.resetTransform()
        self.scale(new_zoom, new_zoom)
        self.setTransformationAnchor(QGraphicsView.NoAnchor)

        self.viewer.zoom_lbl.setText(f"zoom  {new_zoom:.2f}×")

    # ── pan (middle button) ───────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._pan_last = event.pos()
            self.setCursor(QCursor(Qt.ClosedHandCursor))
        elif event.button() == Qt.LeftButton:
            self._sel_start   = self.mapToScene(event.pos())
            self._sel_current = self._sel_start
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan_last is not None:
            delta = event.pos() - self._pan_last
            self._pan_last = event.pos()
            # translate scene visible area by -delta (in scene units)
            h = self.horizontalScrollBar()
            v = self.verticalScrollBar()
            h.setValue(h.value() - delta.x())
            v.setValue(v.value() - delta.y())

        elif self._sel_start is not None:
            self._sel_current = self.mapToScene(event.pos())
            self.viewport().update()

        # FIX 8: update hover_sel on every move
        scene_pos = self.mapToScene(event.pos())
        hit = None
        for sel in reversed(self.viewer.selections):
            x0, y0, x1, y1 = sel["img_bbox"]
            if x0 <= scene_pos.x() <= x1 and y0 <= scene_pos.y() <= y1:
                hit = sel; break
        if hit is not self.viewer.hover_sel:
            self.viewer.hover_sel = hit
            self.viewport().update()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._pan_last = None
            self.setCursor(QCursor(Qt.ArrowCursor))

        elif event.button() == Qt.LeftButton and self._sel_start is not None:
            end = self.mapToScene(event.pos())
            x0 = min(self._sel_start.x(), end.x())
            y0 = min(self._sel_start.y(), end.y())
            x1 = max(self._sel_start.x(), end.x())
            y1 = max(self._sel_start.y(), end.y())
            self.viewer.add_selection(x0, y0, x1, y1)
            self._sel_start   = None
            self._sel_current = None
            self.viewport().update()

        elif event.button() == Qt.RightButton:
            pos = self.mapToScene(event.pos())
            for sel in reversed(self.viewer.selections):
                x0, y0, x1, y1 = sel["img_bbox"]
                if x0 <= pos.x() <= x1 and y0 <= pos.y() <= y1:
                    self.viewer._delete_sel(sel)
                    break

        super().mouseReleaseEvent(event)

    # ── drawing ───────────────────────────────────────────────────────────────

    def drawForeground(self, painter: QPainter, rect):
        """
        Draw selection overlays in SCENE coordinates.
        FIX 3: use scene-space QRectF, not mapFromScene (screen coords).
        """
        super().drawForeground(painter, rect)
        painter.setRenderHint(QPainter.Antialiasing, False)

        # Live drag rectangle
        if self._sel_start is not None and self._sel_current is not None:
            r = QRectF(self._sel_start, self._sel_current).normalized()
            pen = QPen(self.viewer.SEL_COLOR, self.viewer.SEL_WIDTH / self.transform().m11(),
                       Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(r)

        # Committed selections
        font = QFont("Courier New", max(1, int(9 / self.transform().m11())))
        font.setBold(True)
        painter.setFont(font)

        for sel in self.viewer.selections:
            x0, y0, x1, y1 = sel["img_bbox"]
            r = QRectF(x0, y0, x1 - x0, y1 - y0)

            is_hover = (sel is self.viewer.hover_sel)
            color    = (self.viewer.HOVER_CLR  if is_hover else
                        self.viewer.SAVED_CLR  if sel["saved"] else
                        self.viewer.PENDING_CLR)
            lw = (3 if is_hover else 2) / self.transform().m11()

            pen = QPen(color, lw)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(r)

            # Label (offset 3 px at current zoom)
            pad = 3 / self.transform().m11()
            painter.setPen(QPen(color))
            painter.drawText(QPointF(x0 + pad, y0 + pad + font.pointSize()), sel["name"] or "?")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def launch_viewer(image_path: str, symbol_folder: str):
    app = QApplication.instance() or QApplication(sys.argv)
    window = SymbolViewer(image_path, symbol_folder)
    window.showMaximized()
    app.exec_()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    launch_viewer(sys.argv[1], sys.argv[2])