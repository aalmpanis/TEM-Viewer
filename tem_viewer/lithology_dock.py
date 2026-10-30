import math

import numpy as np
from qgis.PyQt.QtWidgets import (
    QDockWidget, QWidget, QScrollArea, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton
)
from qgis.PyQt.QtCore import Qt, QRect, pyqtSignal
from qgis.PyQt.QtGui import QPainter, QFont, QFontMetrics, QColor, QPen, QPalette
from .tv_compat import qflags
from .borehole_section import lith_rgb


class LithologyDockWidget(QDockWidget):
    """Right-hand panel with borehole logs and sounding models side by side
    (same depth scale for all columns)."""
    paramChanged = pyqtSignal()
    clearRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("TEM-Viewer Logs", parent)
        self.setObjectName("TEMViewerLogsDock")
        self.setMinimumWidth(260)
        self.setMinimumHeight(300)

        outer = QWidget()
        ov = QVBoxLayout(outer)
        ov.setContentsMargins(4, 4, 4, 4)
        ov.setSpacing(4)
        head = QHBoxLayout()
        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: #555;")
        self.param_label = QLabel("Sounding:")
        self.param_combo = QComboBox()
        self.param_combo.setToolTip("Model parameter shown for the soundings (IP inversions)")
        self.param_combo.currentIndexChanged.connect(lambda *_: self.paramChanged.emit())
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setToolTip("Remove all columns")
        self.clear_btn.clicked.connect(lambda: self.clearRequested.emit())
        head.addWidget(self.info_label, 1)
        head.addWidget(self.param_label)
        head.addWidget(self.param_combo)
        head.addWidget(self.clear_btn)
        self.param_label.setVisible(False)
        self.param_combo.setVisible(False)
        ov.addLayout(head)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.content = QWidget()
        self.cols_layout = QHBoxLayout(self.content)
        self.cols_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.cols_layout.setSpacing(10)
        self.cols_layout.setContentsMargins(6, 6, 6, 6)
        scroll.setWidget(self.content)
        ov.addWidget(scroll, 1)
        self.setWidget(outer)

    def set_params(self, params, labels):
        """Show the parameter selector when IP parameters are available."""
        cur = self.param_combo.currentData()
        self.param_combo.blockSignals(True)
        self.param_combo.clear()
        for p in params:
            self.param_combo.addItem(labels.get(p, p), p)
        k = self.param_combo.findData(cur) if cur else -1
        self.param_combo.setCurrentIndex(k if k >= 0 else 0)
        self.param_combo.blockSignals(False)
        vis = len(params) > 1
        self.param_label.setVisible(vis)
        self.param_combo.setVisible(vis)

    def current_param(self):
        return self.param_combo.currentData() or 'RHO'

    def clear_columns(self):
        for _ in range(int(self.cols_layout.count())):
            item = self.cols_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
            del item

    def set_hint(self, text):
        self.clear_columns()
        hint = QLabel(text)
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setWordWrap(True)
        self.cols_layout.addWidget(hint)

    def show_columns(self, columns_data, global_total_depth=None):
        """
        columns_data: list of dicts
          borehole: {'name', 'intervals', 'max_bottom'}
          sounding: {'kind': 'sounding', 'name', 'subtitle', 'snd': {...}, 'max_bottom'}
        global_total_depth: every column uses this same depth scale, so the depth
                            axes of boreholes and soundings line up.
        """
        self.clear_columns()
        for col in columns_data:
            col_widget = QWidget()
            col_layout = QVBoxLayout(col_widget)
            col_layout.setContentsMargins(0, 0, 0, 0)
            col_layout.setSpacing(2)
            sub = col.get('subtitle')
            title = QLabel(f"<b>{col['name']}</b>" + (f"<br><small>{sub}</small>" if sub else ""))
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title.setFixedHeight(34)
            col_layout.addWidget(title)
            depth = global_total_depth if global_total_depth is not None else (col['max_bottom'] + 10.0)
            if col.get('kind') == 'sounding':
                view = SoundingView()
                view.set_data(col['snd'], depth)
            else:
                view = LithologyView()
                view.set_data(col['intervals'], depth)
            view.setMinimumWidth(190 if col.get('kind') == 'sounding' else 180)
            view.setMinimumHeight(300)
            col_layout.addWidget(view, 1)
            self.cols_layout.addWidget(col_widget)
        self.cols_layout.addStretch(1)

    def show_lithology(self, intervals, borehole_name, total_depth):
        # convenience wrapper for a single column
        self.show_columns([{
            'name': borehole_name,
            'intervals': intervals,
            'max_bottom': total_depth - 10.0,
        }], global_total_depth=total_depth)

    def clear(self):
        self.setWindowTitle("TEM-Viewer Logs")
        self.clear_columns()


def _nice_step(span, target=8):
    raw = span / max(target, 1)
    for s in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000):
        if s >= raw:
            return s
    return 10 ** math.ceil(math.log10(raw))


class SoundingView(QWidget):
    """Sounding model as a coloured column plus a step curve, on the same depth
    scale (and margins) as LithologyView so the two line up."""
    MARGIN_LEFT = 45
    MARGIN_RIGHT = 8
    MARGIN_TOP = 20
    MARGIN_BOTTOM = 20
    BAR_W = 30
    GAP = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self.d = None
        self.total_depth = 1.0
        self.setMinimumHeight(300)
        self.setBackgroundRole(QPalette.ColorRole.Base)
        self.setAutoFillBackground(True)
        self.setMouseTracking(True)

    def set_data(self, d, total_depth):
        self.d = d
        self.total_depth = float(total_depth) if total_depth and total_depth > 0 else 1.0
        self.update()

    # geometry helpers
    def _y(self, depth):
        total_h = max(1, self.height() - self.MARGIN_TOP - self.MARGIN_BOTTOM)
        return self.MARGIN_TOP + (min(max(depth, 0.0), self.total_depth) / self.total_depth) * total_h

    def _curve_box(self):
        x0 = self.MARGIN_LEFT + self.BAR_W + self.GAP
        x1 = self.width() - self.MARGIN_RIGHT
        return x0, max(x0 + 20, x1)

    def _xv(self, v):
        d = self.d
        x0, x1 = self._curve_box()
        if d['log']:
            if not v > 0:
                return None
            t = (math.log10(v) - math.log10(d['vmin'])) / (math.log10(d['vmax']) - math.log10(d['vmin']))
        else:
            t = (v - d['vmin']) / (d['vmax'] - d['vmin'])
        return x0 + min(max(t, 0.0), 1.0) * (x1 - x0)

    def _color(self, v):
        d = self.d
        if v is None or not np.isfinite(v) or (d['log'] and v <= 0):
            return QColor(235, 235, 235)
        if d['log']:
            t = (math.log10(v) - math.log10(d['vmin'])) / (math.log10(d['vmax']) - math.log10(d['vmin']))
        else:
            t = (v - d['vmin']) / (d['vmax'] - d['vmin'])
        r, g, b, _ = d['cmap'](min(max(t, 0.0), 1.0))
        return QColor(int(r * 255), int(g * 255), int(b * 255))

    def mouseMoveEvent(self, event):
        d = self.d
        if d is None:
            return
        my = event.pos().y()
        total_h = max(1, self.height() - self.MARGIN_TOP - self.MARGIN_BOTTOM)
        depth = (my - self.MARGIN_TOP) / total_h * self.total_depth
        j = np.where((d['top'] <= depth) & (depth < d['bot']))[0]
        if not len(j):
            self.setToolTip("")
            return
        j = int(j[0])
        v = d['vals'][j]
        val = f"{v:.4g} {d['unit']}".strip() if np.isfinite(v) else "–"
        tip = (f"<b>Layer {j + 1}</b><br>{d['top'][j]:.1f} – {d['bot'][j]:.1f} m<br>"
               f"{d['label']}: {val}")
        if np.isfinite(d['doi']) and depth > d['doi']:
            tip += "<br><i>below DOI</i>"
        self.setToolTip(tip)

    def paintEvent(self, event):
        p = QPainter(self)
        if not p.isActive():
            return
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(255, 255, 255))
        d = self.d
        if d is None:
            p.end()
            return
        top_y, bot_y = self._y(0.0), self._y(self.total_depth)
        bx = self.MARGIN_LEFT
        x0, x1 = self._curve_box()
        font = QFont("Arial", 7)
        p.setFont(font)
        fm = QFontMetrics(font)

        # depth ticks
        step = _nice_step(self.total_depth)
        p.setPen(QPen(QColor(0, 0, 0), 1))
        v = 0.0
        while v <= self.total_depth + 1e-9:
            y = self._y(v)
            p.drawLine(int(bx - 4), int(y), int(bx), int(y))
            p.drawText(0, int(y) - 8, bx - 6, 16,
                       qflags(Qt.AlignmentFlag.AlignRight, Qt.AlignmentFlag.AlignVCenter), f"{v:g}")
            p.setPen(QPen(QColor(225, 225, 225), 1))
            p.drawLine(int(x0), int(y), int(x1), int(y))
            p.setPen(QPen(QColor(0, 0, 0), 1))
            v += step

        # coloured column
        for j in range(len(d['top'])):
            t, b = d['top'][j], d['bot'][j]
            if not (np.isfinite(t) and np.isfinite(b)) or b <= t or t >= self.total_depth:
                continue
            y1, y2 = self._y(t), self._y(b)
            p.fillRect(int(bx), int(y1), self.BAR_W, max(1, int(round(y2)) - int(y1)), self._color(d['vals'][j]))
        p.setPen(QPen(QColor(90, 90, 90), 1))
        p.drawRect(int(bx), int(top_y), self.BAR_W, int(bot_y - top_y))

        # curve frame + x ticks
        p.drawRect(int(x0), int(top_y), int(x1 - x0), int(bot_y - top_y))
        ticks = []
        if d['log']:
            e0, e1 = math.floor(math.log10(d['vmin'])), math.ceil(math.log10(d['vmax']))
            ticks = [10.0 ** e for e in range(e0, e1 + 1) if d['vmin'] <= 10.0 ** e <= d['vmax']]
            if len(ticks) < 2:
                ticks = [d['vmin'], d['vmax']]
        else:
            ticks = list(np.linspace(d['vmin'], d['vmax'], 3))
        for tv in ticks:
            x = self._xv(tv)
            if x is None:
                continue
            p.setPen(QPen(QColor(225, 225, 225), 1))
            p.drawLine(int(x), int(top_y), int(x), int(bot_y))
            p.setPen(QPen(QColor(0, 0, 0), 1))
            lab = f"{tv:g}" if 1e-3 <= abs(tv) < 1e5 else f"{tv:.2g}"
            p.drawText(int(x - 30), int(bot_y) + 2, 60, 14, qflags(Qt.AlignmentFlag.AlignHCenter,
                                                                   Qt.AlignmentFlag.AlignTop), lab)
        p.drawText(int(x0), 2, int(x1 - x0), 16, qflags(Qt.AlignmentFlag.AlignHCenter,
                                                        Qt.AlignmentFlag.AlignVCenter),
                   f"{d['label']}" + (f" ({d['unit']})" if d['unit'] else ""))

        # step curve
        pen = QPen(QColor(20, 20, 20), 2)
        p.setPen(pen)
        prev = None
        for j in range(len(d['top'])):
            t, b, v = d['top'][j], d['bot'][j], d['vals'][j]
            if not (np.isfinite(t) and np.isfinite(b)) or t >= self.total_depth:
                continue
            x = self._xv(v) if np.isfinite(v) else None
            if x is None:
                prev = None
                continue
            y1, y2 = self._y(t), self._y(b)
            if prev is not None:
                p.drawLine(int(prev[0]), int(prev[1]), int(x), int(y1))
            p.drawLine(int(x), int(y1), int(x), int(y2))
            prev = (x, y2)

        # DOI
        if np.isfinite(d['doi']) and d['doi'] < self.total_depth:
            yd = self._y(d['doi'])
            p.fillRect(int(bx), int(yd), int(x1 - bx), int(bot_y - yd), QColor(255, 255, 255, 140))
            pen = QPen(QColor(200, 0, 0), 1)
            pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawLine(int(bx), int(yd), int(x1), int(yd))
            p.drawText(int(x1 - 30), int(yd) - 14, 28, 12,
                       qflags(Qt.AlignmentFlag.AlignRight, Qt.AlignmentFlag.AlignVCenter), "DOI")
        p.end()


class LithologyView(QWidget):
    MARGIN_LEFT = 45
    MARGIN_RIGHT = 10
    MARGIN_TOP = 20
    MARGIN_BOTTOM = 20
    BAR_WIDTH = 140

    def __init__(self, parent=None):
        super().__init__(parent)
        self.intervals = []
        self.total_depth = 1.0
        self.hint = ""
        self.setMinimumHeight(300)
        self.setBackgroundRole(QPalette.ColorRole.Base)
        self.setAutoFillBackground(True)
        self.setMouseTracking(True)

    def set_data(self, intervals, total_depth):
        self.intervals = intervals
        self.total_depth = float(total_depth) if total_depth and total_depth > 0 else 1.0
        self.hint = ""
        self.update()

    def clear_data(self):
        self.intervals = []
        self.total_depth = 1.0
        self.hint = ""
        self.update()

    def set_hint(self, text):
        self.hint = text
        self.intervals = []
        self.total_depth = 1.0
        self.update()

    # -----------------------------------------------------------------
    # Tooltip on hover
    # -----------------------------------------------------------------
    def mouseMoveEvent(self, event):
        if self.hint or not self.intervals or self.total_depth <= 0:
            self.setToolTip("")
            return

        mx = event.pos().x()
        my = event.pos().y()

        bar_w = min(self.BAR_WIDTH,
                    self.width() - self.MARGIN_LEFT - self.MARGIN_RIGHT)
        bar_x = self.MARGIN_LEFT
        if mx < bar_x or mx > bar_x + bar_w:
            self.setToolTip("")
            return

        total_h = max(1, self.height() - self.MARGIN_TOP - self.MARGIN_BOTTOM)

        for interval in self.intervals:
            top = float(interval['top'])
            bottom = float(interval['bottom'])
            if bottom <= top:
                continue

            y1 = self.MARGIN_TOP + (top / self.total_depth) * total_h
            y2 = self.MARGIN_TOP + (bottom / self.total_depth) * total_h

            if y1 <= my <= y2:
                tip = (
                    f"<b>{interval['general']}</b><br>"
                    f"<i>{interval['detail']}</i><br>"
                    f"Top: {top:.2f} m<br>"
                    f"Bottom: {bottom:.2f} m"
                )
                self.setToolTip(tip)
                return

        self.setToolTip("")

    # -----------------------------------------------------------------

    def _color_for(self, lith_name):
        return QColor(*lith_rgb(lith_name))

    def paintEvent(self, event):
        painter = QPainter(self)
        if not painter.isActive():
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.fillRect(self.rect(), QColor(255, 255, 255))

        if self.hint:
            painter.setPen(QColor(80, 80, 80))
            painter.setFont(QFont("Arial", 10))
            painter.drawText(self.rect(), qflags(Qt.AlignmentFlag.AlignCenter, Qt.TextFlag.TextWordWrap), self.hint)
            painter.end()
            return

        if not self.intervals:                 # borehole without a log (yet): empty column
            w, h = self.width(), self.height()
            bar_w = min(self.BAR_WIDTH, w - self.MARGIN_LEFT - self.MARGIN_RIGHT)
            total_h = max(1, h - self.MARGIN_TOP - self.MARGIN_BOTTOM)
            pen = QPen(QColor(150, 150, 150), 1)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(int(self.MARGIN_LEFT), int(self.MARGIN_TOP), int(bar_w), int(total_h))
            painter.setPen(QColor(120, 120, 120))
            painter.setFont(QFont("Arial", 9))
            painter.drawText(int(self.MARGIN_LEFT), int(self.MARGIN_TOP), int(bar_w), int(total_h),
                             qflags(Qt.AlignmentFlag.AlignCenter, Qt.TextFlag.TextWordWrap),
                             "No lithology log yet")
            painter.end()
            return

        w = self.width()
        h = self.height()
        bar_w = min(self.BAR_WIDTH, w - self.MARGIN_LEFT - self.MARGIN_RIGHT)
        bar_x = self.MARGIN_LEFT
        total_h = max(1, h - self.MARGIN_TOP - self.MARGIN_BOTTOM)

        # Outer frame
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawRect(int(bar_x), int(self.MARGIN_TOP), int(bar_w), int(total_h))

        # Fonts
        font_bold = QFont("Arial", 9)
        font_bold.setBold(True)
        font_small = QFont("Arial", 7)
        font_label = QFont("Arial", 8)

        for interval in self.intervals:
            top = float(interval['top'])
            bottom = float(interval['bottom'])
            if bottom <= top:
                continue

            y1 = self.MARGIN_TOP + (top / self.total_depth) * total_h
            y2 = self.MARGIN_TOP + (bottom / self.total_depth) * total_h
            rect_h = max(1, y2 - y1)

            # Fill colour
            color = self._color_for(interval['general'])
            painter.fillRect(int(bar_x), int(y1), int(bar_w), int(rect_h), color)
            painter.setPen(QPen(QColor(80, 80, 80), 1))
            painter.drawRect(int(bar_x), int(y1), int(bar_w), int(rect_h))

            # Depth labels left
            painter.setPen(QColor(0, 0, 0))
            painter.setFont(font_label)
            painter.drawText(
                0, int(y1) - 10, self.MARGIN_LEFT - 5, 20,
                qflags(Qt.AlignmentFlag.AlignRight, Qt.AlignmentFlag.AlignVCenter), f"{top:g}"
            )
            painter.drawText(
                0, int(y2) - 10, self.MARGIN_LEFT - 5, 20,
                qflags(Qt.AlignmentFlag.AlignRight, Qt.AlignmentFlag.AlignVCenter), f"{bottom:g}"
            )

            # Text inside block
            pad = 4
            tx = int(bar_x + pad)
            ty = int(y1 + pad)
            tw = int(bar_w - pad * 2)
            th = int(rect_h - pad * 2)
            if tw > 10 and th > 10:
                text_rect = QRect(tx, ty, tw, th)
                painter.save()
                painter.setClipRect(text_rect)

                # General lithology – bold (white text on dark colours)
                dark = (0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()) < 120
                painter.setPen(QColor(255, 255, 255) if dark else QColor(0, 0, 0))
                painter.setFont(font_bold)
                flags = qflags(Qt.AlignmentFlag.AlignHCenter, Qt.AlignmentFlag.AlignTop, Qt.TextFlag.TextWordWrap)
                fm_bold = QFontMetrics(font_bold)
                general_h = fm_bold.boundingRect(text_rect, flags, interval['general']).height()
                painter.drawText(text_rect, flags, interval['general'])

                # Detailed description – small
                detail_rect = QRect(tx, ty + general_h + 2, tw, max(0, th - general_h - 2))
                painter.setFont(font_small)
                painter.setPen(QColor(230, 230, 230) if dark else QColor(60, 60, 60))
                painter.drawText(detail_rect, flags, interval['detail'])

                painter.restore()

        painter.end()