# -*- coding: utf-8 -*-
"""
TEM-Viewer - small colour bar drawn with QPainter.

Used twice by the Soundings tab:
  * horizontally inside the tab (preview of the current colour settings), and
  * vertically as an overlay in a corner of the QGIS map canvas, so the map
    shows a legend that follows every change of colormap, range or scale.
"""
import numpy as np
import matplotlib.colors
from matplotlib.ticker import MaxNLocator, LogLocator

from qgis.PyQt.QtCore import Qt, QRectF, QEvent, QObject
from qgis.PyQt.QtGui import QPainter, QColor, QPen, QBrush, QFont, QFontMetrics
from qgis.PyQt.QtWidgets import QWidget

from .tv_compat import qflags


def colorbar_spec(cmap, lo, hi, log, n_classes, title):
    """Everything the widget needs: class colours (equal steps in linear or log space)
    and tick values/positions (0..1 along the bar)."""
    if hi <= lo:
        hi = lo + 1.0
    log = bool(log and lo > 0)
    norm = matplotlib.colors.LogNorm(lo, hi) if log else matplotlib.colors.Normalize(lo, hi)
    colors = []
    for k in range(n_classes):
        t = (k + 0.5) / n_classes
        r, g, b, _ = cmap(t)
        colors.append(QColor(int(r * 255), int(g * 255), int(b * 255)))
    if log:
        def _log_ticks(subs):
            return [v for v in LogLocator(base=10, subs=subs, numticks=50).tick_values(lo, hi)
                    if lo <= v <= hi]
        ticks = _log_ticks((1.0, 2.0, 5.0))
        if len(ticks) > 6:
            ticks = _log_ticks((1.0,))
        elif len(ticks) < 3:                      # narrow range: 1..9 steps, thinned
            ticks = _log_ticks(tuple(float(k) for k in range(1, 10)))
            while len(ticks) > 5:
                ticks = ticks[::2]
    else:
        ticks = [v for v in MaxNLocator(5).tick_values(lo, hi) if lo <= v <= hi]
    ticks = sorted(set([lo, hi] + ticks))
    pos = [float(norm(v)) for v in ticks]
    # drop inner ticks that would collide with the end labels
    keep_t, keep_p = [], []
    for v, p in zip(ticks, pos):
        if v not in (lo, hi) and (p < 0.08 or p > 0.92):
            continue
        keep_t.append(v)
        keep_p.append(p)
    return dict(colors=colors, ticks=keep_t, pos=keep_p, title=title)


def _fmt(v):
    a = abs(v)
    if a != 0 and (a < 1e-3 or a >= 1e5):
        return f"{v:.2g}"
    return f"{v:.4g}"


class ColorbarWidget(QWidget):
    """horizontal=True: bar across the widget with labels below (tab preview).
    horizontal=False: compact vertical legend with a title (map overlay)."""

    def __init__(self, parent=None, horizontal=True):
        super().__init__(parent)
        self.horizontal = horizontal
        self.spec = None
        if horizontal:
            self.setMinimumHeight(44)
            self.setMaximumHeight(44)
        else:
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def set_spec(self, spec):
        self.spec = spec
        if not self.horizontal:
            self._fit_size()
        self.update()

    # -- vertical overlay size from the labels ----------------------------
    def _fit_size(self):
        if not self.spec:
            return
        fm = QFontMetrics(self._font())
        tw = max([fm.horizontalAdvance(_fmt(v)) for v in self.spec['ticks']] + [0])
        title_w = fm.horizontalAdvance(self.spec['title']) + 16
        w = max(14 + 6 + 4 + tw + 16, title_w)
        h = 16 + fm.height() + 6 + 170 + fm.height() // 2 + 10
        self.resize(int(w), int(h))

    def _font(self):
        f = QFont(self.font())
        f.setPointSizeF(8.0)
        return f

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setFont(self._font())
        fm = QFontMetrics(p.font())
        if not self.spec:
            p.setPen(QColor(120, 120, 120))
            p.drawText(self.rect(), qflags(Qt.AlignmentFlag.AlignCenter),
                       "Choose a layer and attribute to see the colour bar")
            p.end()
            return
        cols = self.spec['colors']
        n = len(cols)
        black = QPen(QColor(40, 40, 40))
        black.setWidthF(0.8)
        if self.horizontal:
            x0, x1 = 8.0, float(self.width() - 8)
            y0, bh = 4.0, 14.0
            w = (x1 - x0) / n
            p.setPen(Qt.PenStyle.NoPen)
            for k, c in enumerate(cols):
                p.setBrush(QBrush(c))
                p.drawRect(QRectF(x0 + k * w, y0, w + 0.6, bh))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(black)
            p.drawRect(QRectF(x0, y0, x1 - x0, bh))
            for v, t in zip(self.spec['ticks'], self.spec['pos']):
                x = x0 + t * (x1 - x0)
                p.drawLine(int(round(x)), int(y0 + bh), int(round(x)), int(y0 + bh + 3))
                lab = _fmt(v)
                lw = fm.horizontalAdvance(lab)
                lx = min(max(x - lw / 2.0, 0.0), self.width() - lw)
                p.drawText(int(lx), int(y0 + bh + 4 + fm.ascent()), lab)
            p.end()
            return

        # vertical overlay with a light background box
        p.setPen(QPen(QColor(120, 120, 120)))
        p.setBrush(QBrush(QColor(255, 255, 255, 225)))
        p.drawRoundedRect(QRectF(0.5, 0.5, self.width() - 1.0, self.height() - 1.0), 4, 4)
        p.setPen(black)
        f = self._font()
        f.setBold(True)
        p.setFont(f)
        p.drawText(8, 6 + fm.ascent(), self.spec['title'])
        p.setFont(self._font())
        top = 6 + fm.height() + 8 + fm.height() / 2.0
        bar_h = 170.0
        bx, bw = 10.0, 14.0
        h = bar_h / n
        p.setPen(Qt.PenStyle.NoPen)
        for k, c in enumerate(cols):          # high values at the top
            p.setBrush(QBrush(c))
            p.drawRect(QRectF(bx, top + bar_h - (k + 1) * h, bw, h + 0.6))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(black)
        p.drawRect(QRectF(bx, top, bw, bar_h))
        for v, t in zip(self.spec['ticks'], self.spec['pos']):
            y = top + bar_h - t * bar_h
            p.drawLine(int(bx + bw), int(round(y)), int(bx + bw + 4), int(round(y)))
            p.drawText(int(bx + bw + 7), int(round(y + fm.ascent() / 2.0 - 1)), _fmt(v))
        p.end()


class MapColorbar(QObject):
    """Keeps a vertical ColorbarWidget in a corner of the map canvas, visible only
    while its layer is shown on the map."""
    CORNERS = ("Bottom right", "Bottom left", "Top right", "Top left")

    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.widget = ColorbarWidget(canvas, horizontal=False)
        self.widget.hide()
        self.corner = 0
        self.layer_id = None
        self.enabled = True
        canvas.installEventFilter(self)
        try:
            canvas.mapCanvasRefreshed.connect(self._sync_visibility)
            canvas.layersChanged.connect(self._sync_visibility)
        except Exception:
            pass

    def eventFilter(self, obj, ev):
        if obj is self.canvas and ev.type() == QEvent.Type.Resize:
            self._place()
        return False

    def set(self, layer_id, spec):
        self.layer_id = layer_id
        self.widget.set_spec(spec)
        self._place()
        self._sync_visibility()

    def set_corner(self, idx):
        self.corner = int(idx)
        self._place()

    def set_enabled(self, on):
        self.enabled = bool(on)
        self._sync_visibility()

    def _place(self):
        m = 10
        w, h = self.widget.width(), self.widget.height()
        cw, ch = self.canvas.width(), self.canvas.height()
        x = cw - w - m if self.corner in (0, 2) else m
        y = ch - h - m if self.corner in (0, 1) else m
        self.widget.move(int(x), int(y))

    def _sync_visibility(self, *args):
        show = False
        if self.enabled and self.layer_id and self.widget.spec:
            try:
                show = any(l.id() == self.layer_id for l in self.canvas.layers())
            except Exception:
                show = False
        if show:
            self._place()
            self.widget.show()
            self.widget.raise_()
        else:
            self.widget.hide()

    def remove(self):
        try:
            self.canvas.removeEventFilter(self)
        except Exception:
            pass
        for sig in ('mapCanvasRefreshed', 'layersChanged'):
            try:
                getattr(self.canvas, sig).disconnect(self._sync_visibility)
            except Exception:
                pass
        try:
            self.widget.hide()
            self.widget.setParent(None)
            self.widget.deleteLater()
        except Exception:
            pass
