# -*- coding: utf-8 -*-
"""
TEM-Viewer - Mean Resistivity (MR) maps.

Depth (or elevation) slices of the 1D models (resistivity, or for IP inversions
also phase PHI, tau TPHI and C), averaged per sounding
over each slice and interpolated to a regular grid with inverse distance
weighting or ordinary kriging. Shown in the tab and, on request, added to the
QGIS map as GeoTIFF layers.
"""
import os
import re
import math
import tempfile

import numpy as np
import pandas as pd

from qgis.PyQt.QtCore import Qt, QTimer
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel, QLineEdit,
    QPushButton, QComboBox, QCheckBox, QDoubleSpinBox, QSpinBox, QSplitter,
    QScrollArea, QFileDialog, QMessageBox
)
from qgis.core import (QgsProject, QgsRasterLayer, Qgis, QgsVectorLayer, QgsGeometry,
                       QgsCoordinateTransform, QgsRectangle, QgsPointXY)
from qgis.gui import QgsRubberBand

from .tv_compat import (FigureCanvasQTAgg as FigureCanvas,
                             NavigationToolbar2QT as NavigationToolbar, qgis_enum)
import matplotlib
import matplotlib.colors
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter

from .tv_data import LayerPicker
from .resistivity_tool import RESISTIVITY_CMAPS, get_colormap, PARAMS, PARAM_ORDER, available_params


# =============================================================================
#  Computation (no Qt)
# =============================================================================
def _numbered(cols, prefix):
    out = [c for c in cols if re.match(rf'{prefix}\d+$', str(c))]
    return sorted(out, key=lambda c: int(re.findall(r'\d+$', c)[0]))


def model_arrays(df, param='RHO'):
    """values, top, bottom (n_soundings x n_layers, depths in m below surface).
    param: 'RHO' (resistivity) or an IP parameter ('PHI', 'TPHI', 'C')."""
    rho_c = _numbered(df.columns, 'RHO_')
    top_c = _numbered(df.columns, 'DEP_TOP_')
    bot_c = _numbered(df.columns, 'DEP_BOT_')
    thk_c = _numbered(df.columns, 'THK_')
    if not rho_c:
        raise ValueError("No RHO_i columns in the sounding data.")
    n, L = len(df), len(rho_c)
    val_c = rho_c if param == 'RHO' else _numbered(df.columns, param + '_')[:L]
    if not val_c:
        raise ValueError(f"No {param}_i columns in the sounding data.")
    rho = np.full((n, L), np.nan)
    rho[:, :len(val_c)] = df[val_c].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float, copy=True)
    if len(top_c) >= L:
        top = df[top_c[:L]].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float, copy=True)
    elif thk_c:
        thk = df[thk_c].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float, copy=True)
        top = np.zeros((n, L))
        top[:, 1:] = np.cumsum(thk[:, :L - 1], axis=1)
    else:
        raise ValueError("No DEP_TOP_i or THK_i columns to define the layer depths.")
    bot = np.full((n, L), np.inf)
    if bot_c:
        b = df[bot_c].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float, copy=True)
        bot[:, :b.shape[1]] = b[:, :L]
    bot[:, :-1] = np.where(np.isfinite(bot[:, :-1]), bot[:, :-1], top[:, 1:])
    if thk_c and len(thk_c) >= L:
        thk_last = pd.to_numeric(df[thk_c[L - 1]], errors='coerce').to_numpy(dtype=float, copy=True)
        bot[:, -1] = np.where(np.isfinite(bot[:, -1]), bot[:, -1], top[:, -1] + thk_last)
    rho[(rho <= 0)] = np.nan
    return rho, top, bot


def slice_means(df, intervals, reference='depth', average='log', doi_col=None, param='RHO'):
    """Mean resistivity (or IP parameter) per sounding and slice.

    intervals : list of (a, b); depth below surface (a < b) or elevation (a > b)
    average   : 'log' (geometric), 'arith' or 'cond' (mean conductivity)
    doi_col   : only use the model above this DOI column (None = whole model)
    Returns array n_soundings x n_slices (NaN where there is no model).
    """
    rho, top, bot = model_arrays(df, param)
    elev = pd.to_numeric(df['ELEVATION'], errors='coerce').to_numpy(dtype=float, copy=True)
    doi = None
    if doi_col and doi_col in df.columns:
        doi = pd.to_numeric(df[doi_col], errors='coerce').to_numpy(dtype=float)
    out = np.full((len(df), len(intervals)), np.nan)
    valid = np.isfinite(rho)
    with np.errstate(divide='ignore', invalid='ignore'):
        logr = np.log10(rho)
        for k, (a, b) in enumerate(intervals):
            if reference == 'depth':
                d0 = np.full(len(df), float(min(a, b)))
                d1 = np.full(len(df), float(max(a, b)))
            else:                         # absolute elevation, a = upper, b = lower
                hi, lo = max(a, b), min(a, b)
                d0 = np.clip(elev - hi, 0, None)
                d1 = elev - lo
            if doi is not None:
                d1 = np.where(np.isfinite(doi), np.minimum(d1, doi), d1)
            ov = np.clip(np.minimum(bot, d1[:, None]) - np.maximum(top, d0[:, None]), 0, None)
            ov = np.where(valid, ov, 0.0)
            w = ov.sum(axis=1)
            ok = w > 0
            if average == 'log':
                m = 10 ** (np.nansum(np.where(ov > 0, ov * logr, 0.0), axis=1) / w)
            elif average == 'arith':
                m = np.nansum(ov * rho, axis=1) / w
            else:
                m = 1.0 / (np.nansum(ov / rho, axis=1) / w)
            out[ok, k] = m[ok]
    return out


def make_grid(x, y, spacing, margin):
    x0, x1 = np.nanmin(x) - margin, np.nanmax(x) + margin
    y0, y1 = np.nanmin(y) - margin, np.nanmax(y) + margin
    nx = int(math.floor((x1 - x0) / spacing)) + 1
    ny = int(math.floor((y1 - y0) / spacing)) + 1
    xs = x0 + spacing * np.arange(nx)
    ys = y0 + spacing * np.arange(ny)
    return xs, ys


def _neighbours(xy, nodes, k, radius):
    """Indices (n_nodes x k) and distances of the k nearest points within radius
    (missing neighbours: index -1, distance inf)."""
    k = min(k, len(xy))
    try:
        from scipy.spatial import cKDTree
        d, i = cKDTree(xy).query(nodes, k=k, distance_upper_bound=radius)
        if k == 1:
            d, i = d[:, None], i[:, None]
        i = np.where(np.isfinite(d), i, -1)
        return i, d
    except ImportError:
        I = np.full((len(nodes), k), -1)
        D = np.full((len(nodes), k), np.inf)
        for s in range(0, len(nodes), 2000):
            nd = nodes[s:s + 2000]
            dd = np.hypot(nd[:, None, 0] - xy[None, :, 0], nd[:, None, 1] - xy[None, :, 1])
            idx = np.argsort(dd, axis=1)[:, :k]
            dk = np.take_along_axis(dd, idx, axis=1)
            idx = np.where(dk <= radius, idx, -1)
            I[s:s + 2000], D[s:s + 2000] = idx, np.where(dk <= radius, dk, np.inf)
        return I, D


def idw(xy, v, nodes, radius, power=2.0, kmax=12):
    idx, d = _neighbours(xy, nodes, kmax, radius)
    has = idx >= 0
    vals = np.where(has, v[np.clip(idx, 0, None)], 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        w = np.where(has, 1.0 / np.maximum(d, 1e-9) ** power, 0.0)
        z = (w * vals).sum(axis=1) / w.sum(axis=1)
    z[~has.any(axis=1)] = np.nan
    exact = has[:, 0] & (d[:, 0] < 1e-6)
    z[exact] = vals[exact, 0]
    return z


# ---- variogram / kriging ----------------------------------------------------
def variogram_model(name):
    def sph(h, n, s, a):
        r = np.clip(h / a, 0, 1)
        return np.where(h > 0, n + s * (1.5 * r - 0.5 * r ** 3), 0.0)

    def exp_(h, n, s, a):
        return np.where(h > 0, n + s * (1 - np.exp(-3 * h / a)), 0.0)

    def gau(h, n, s, a):
        return np.where(h > 0, n + s * (1 - np.exp(-3 * (h / a) ** 2)), 0.0)
    return {'spherical': sph, 'exponential': exp_, 'gaussian': gau}[name]


def fit_variogram(xy, v, model='spherical', fixed_range=None, max_pts=1500, nbins=15, seed=0):
    """Fit nugget, partial sill and range to the empirical semivariogram."""
    rng = np.random.default_rng(seed)
    if len(v) > max_pts:
        sel = rng.choice(len(v), max_pts, replace=False)
        xy, v = xy[sel], v[sel]
    i, j = np.triu_indices(len(v), 1)
    h = np.hypot(*(xy[i] - xy[j]).T)
    g = 0.5 * (v[i] - v[j]) ** 2
    var = float(np.var(v)) if len(v) > 1 else 1.0
    hmax = h.max() / 2 if h.size else 1.0
    if h.size < 10 or hmax <= 0:
        return dict(nugget=0.0, sill=max(var, 1e-6), range=fixed_range or max(hmax, 1.0))
    edges = np.linspace(0, hmax, nbins + 1)
    b = np.digitize(h, edges) - 1
    hc, gc, nc = [], [], []
    for k in range(nbins):
        m = b == k
        if m.sum() >= 3:
            hc.append(h[m].mean()); gc.append(g[m].mean()); nc.append(m.sum())
    hc, gc, nc = map(np.asarray, (hc, gc, nc))
    f = variogram_model(model)
    p0 = [0.1 * var, max(var, 1e-6), fixed_range or hmax / 2]
    try:
        from scipy.optimize import curve_fit
        if fixed_range:
            popt, _ = curve_fit(lambda hh, n, s: f(hh, n, s, fixed_range), hc, gc, p0=p0[:2],
                                bounds=([0, 1e-9], [np.inf, np.inf]), sigma=1 / np.sqrt(nc), maxfev=5000)
            n, s, a = popt[0], popt[1], fixed_range
        else:
            popt, _ = curve_fit(f, hc, gc, p0=p0, bounds=([0, 1e-9, hmax / 50], [np.inf, np.inf, hmax * 4]),
                                sigma=1 / np.sqrt(nc), maxfev=5000)
            n, s, a = popt
    except Exception:
        n, s, a = 0.0, max(var, 1e-6), fixed_range or hmax / 2
    return dict(nugget=float(n), sill=float(s), range=float(a))


def ordinary_kriging(xy, v, nodes, radius, kmax=16, model='spherical', params=None):
    """Local ordinary kriging (k nearest within radius). Nodes with < 3 neighbours
    fall back to inverse distance; nodes without neighbours are NaN."""
    params = params or fit_variogram(xy, v, model)
    f = variogram_model(model)
    n_, s_, a_ = params['nugget'], params['sill'], params['range']
    idx, d = _neighbours(xy, nodes, kmax, radius)
    cnt = (idx >= 0).sum(axis=1)
    z = np.full(len(nodes), np.nan)
    few = (cnt > 0) & (cnt < 3)
    if few.any():
        z[few] = idw(xy, v, nodes[few], radius, 2.0, kmax)
    for c in np.unique(cnt[cnt >= 3]):
        rows = np.where(cnt == c)[0]
        for s in range(0, len(rows), 4000):
            r = rows[s:s + 4000]
            ii = idx[r, :c]
            P = xy[ii]                                         # (m, c, 2)
            hij = np.hypot(P[:, :, None, 0] - P[:, None, :, 0], P[:, :, None, 1] - P[:, None, :, 1])
            A = np.ones((len(r), c + 1, c + 1))
            A[:, :c, :c] = f(hij, n_, s_, a_)
            A[:, c, c] = 0.0
            bvec = np.ones((len(r), c + 1))
            bvec[:, :c] = f(d[r, :c], n_, s_, a_)
            try:
                w = np.linalg.solve(A, bvec[..., None])[..., 0]
            except np.linalg.LinAlgError:
                w = np.einsum('mij,mj->mi', np.linalg.pinv(A), bvec)
            z[r] = (w[:, :c] * v[ii]).sum(axis=1)
    exact = (idx[:, 0] >= 0) & (d[:, 0] < 1e-6)
    z[exact] = v[idx[exact, 0]]
    return z, params


def write_raster(path, Z, xs, ys, spacing, crs_wkt, nodata=-9999.0):
    """Write grid Z[iy, ix] (ys ascending) as GeoTIFF (GDAL) or ESRI ASCII grid.
    Returns the path actually written."""
    data = np.flipud(np.where(np.isfinite(Z), Z, nodata)).astype('float32')
    x_left, y_top = xs[0] - spacing / 2.0, ys[-1] + spacing / 2.0
    try:
        from osgeo import gdal, osr
        drv = gdal.GetDriverByName('GTiff')
        ds = drv.Create(path, data.shape[1], data.shape[0], 1, gdal.GDT_Float32, ['COMPRESS=LZW'])
        ds.SetGeoTransform((x_left, spacing, 0, y_top, 0, -spacing))
        srs = osr.SpatialReference()
        srs.ImportFromWkt(crs_wkt)
        ds.SetProjection(srs.ExportToWkt())
        band = ds.GetRasterBand(1)
        band.WriteArray(data)
        band.SetNoDataValue(nodata)
        band.FlushCache()
        ds = None
        return path
    except ImportError:
        path = os.path.splitext(path)[0] + '.asc'
        with open(path, 'w') as fh:
            fh.write(f"ncols {data.shape[1]}\nnrows {data.shape[0]}\n"
                     f"xllcorner {x_left}\nyllcorner {ys[0] - spacing / 2.0}\n"
                     f"cellsize {spacing}\nNODATA_value {nodata}\n")
            np.savetxt(fh, data, fmt='%.6g')
        with open(os.path.splitext(path)[0] + '.prj', 'w') as fh:
            fh.write(crs_wkt)
        return path


def polygon_mask(rings, nodes):
    """rings: list of (exterior, [holes]) as Nx2 arrays. True where a node is inside."""
    inside = np.zeros(len(nodes), dtype=bool)
    for ext, holes in rings:
        m = Path(ext).contains_points(nodes)
        for h in holes:
            m &= ~Path(h).contains_points(nodes)
        inside |= m
    return inside


# =============================================================================
#  Tab
# =============================================================================
class MRMapsWidget(QWidget):
    def __init__(self, parent=None, iface=None, registry=None):
        super().__init__(parent)
        self.iface = iface
        self.registry = registry
        self.slices = []           # dicts: label, a, b, Z (rho grid), xs, ys, pts, vals
        self.cur = 0
        self.crs = None
        self.spacing = None
        self._drawn_rect = None    # QgsRectangle in canvas CRS
        self._area_tool = None
        self._prev_map_tool = None
        self._area_rb = None
        self._params_avail = ['RHO']
        self._cs = {}              # colour scale per parameter
        self._cs_param = 'RHO'     # parameter the colour widgets show
        self._setup_ui()

    # ------------------------------------------------------------------
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        split = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(split, 1)

        # ---- plot ----
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(10, 6))
        self.ax = self.ax_depth = None
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(380, 320)
        self.toolbar = NavigationToolbar(self.canvas, self)
        lv.addWidget(self.toolbar)
        lv.addWidget(self.canvas, 1)
        self.cursor_label = QLabel("X:   Y:   ρ: ")
        self.cursor_label.setStyleSheet("font-family: monospace; font-size: 10px;")
        lv.addWidget(self.cursor_label)
        nav = QHBoxLayout()
        self.compute_btn = QPushButton("Compute Slices")
        self.compute_btn.setStyleSheet("font-weight: bold;")
        self.compute_btn.clicked.connect(self.compute)
        self.prev_btn = QPushButton("◀")
        self.next_btn = QPushButton("▶")
        self.slice_combo = QComboBox()
        self.slice_combo.setMinimumWidth(170)
        self.prev_btn.clicked.connect(lambda: self._step(-1))
        self.next_btn.clicked.connect(lambda: self._step(1))
        self.slice_combo.currentIndexChanged.connect(self._show_slice)
        self.show_pts_check = QCheckBox("Soundings")
        self.show_pts_check.setChecked(True)
        self.show_pts_check.toggled.connect(lambda *_: self._render())
        self.add_btn = QPushButton("Add Slice to Map")
        self.add_btn.clicked.connect(lambda: self._add_to_map(all_slices=False))
        self.add_all_btn = QPushButton("Add All to Map")
        self.add_all_btn.clicked.connect(lambda: self._add_to_map(all_slices=True))
        for w in (self.compute_btn, self.prev_btn, self.slice_combo, self.next_btn,
                  self.show_pts_check):
            nav.addWidget(w)
        nav.addStretch()
        nav.addWidget(self.add_btn)
        nav.addWidget(self.add_all_btn)
        lv.addLayout(nav)
        split.addWidget(left)

        # ---- settings ----
        panel = QWidget()
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(0, 0, 4, 0)

        g1 = QGroupBox("1. Sounding layers")
        v1 = QVBoxLayout(g1)
        self.picker = LayerPicker(self.registry, 'soundings', self, max_height=80)
        v1.addWidget(self.picker)
        self.param_lbl = QLabel("Parameter:")
        self.param_combo = QComboBox()
        self.param_combo.addItem(PARAMS['RHO']['label'], 'RHO')
        self.param_combo.setToolTip("IP inversions: map resistivity or an IP parameter")
        ph = QHBoxLayout()
        ph.addWidget(self.param_lbl)
        ph.addWidget(self.param_combo, 1)
        v1.addLayout(ph)
        pv.addWidget(g1)

        ga = QGroupBox("Area")
        sa = QGridLayout(ga)
        self.area_combo = QComboBox()
        self.area_combo.addItems(["Whole data extent", "Current map view",
                                  "Rectangle drawn on map", "Polygon layer"])
        self.area_combo.setToolTip("Area for which the maps are made. Soundings up to one "
                                   "search radius outside the area are still used.")
        self.area_combo.currentIndexChanged.connect(self._area_mode_changed)
        self.draw_btn = QPushButton("Draw on map")
        self.draw_btn.setToolTip("Drag a rectangle on the map")
        self.draw_btn.clicked.connect(self._start_draw)
        self.poly_combo = QComboBox()
        self.poly_refresh = QPushButton("↻")
        self.poly_refresh.setMaximumWidth(28)
        self.poly_refresh.clicked.connect(self._refresh_polygons)
        self.poly_sel_check = QCheckBox("Selected polygons only")
        self.area_info = QLabel("")
        self.area_info.setWordWrap(True)
        sa.addWidget(self.area_combo, 0, 0, 1, 2)
        sa.addWidget(self.draw_btn, 0, 2)
        sa.addWidget(self.poly_combo, 1, 0, 1, 2)
        sa.addWidget(self.poly_refresh, 1, 2)
        sa.addWidget(self.poly_sel_check, 2, 0, 1, 3)
        sa.addWidget(self.area_info, 3, 0, 1, 3)
        pv.addWidget(ga)

        g2 = QGroupBox("2. Slices")
        s2 = QGridLayout(g2)
        self.ref_combo = QComboBox()
        self.ref_combo.addItems(["Depth below surface", "Elevation (m a.s.l.)"])
        self.ref_combo.currentIndexChanged.connect(self._ref_changed)
        self.from_spin, self.to_spin, self.thk_spin = QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox()
        for sp, val in ((self.from_spin, 0), (self.to_spin, 100), (self.thk_spin, 10)):
            sp.setRange(-10000, 10000)
            sp.setDecimals(1)
            sp.setValue(val)
            sp.setSuffix(" m")
        self.thk_spin.setRange(0.1, 10000)
        self.avg_combo = QComboBox()
        self.avg_combo.addItems(["Logarithmic (geometric)", "Arithmetic", "Conductivity"])
        self.avg_combo.setToolTip("How the layer values inside a slice are averaged "
                                  "(thickness weighted); conductivity averaging only for resistivity")
        self.doi_check = QCheckBox("Only above DOI")
        self.doi_check.setChecked(True)
        self.doi_combo = QComboBox()
        self.doi_combo.addItems(["Standard", "Conservative"])
        s2.addWidget(QLabel("Reference:"), 0, 0)
        s2.addWidget(self.ref_combo, 0, 1, 1, 3)
        self.from_lbl, self.to_lbl = QLabel("From:"), QLabel("To:")
        s2.addWidget(self.from_lbl, 1, 0)
        s2.addWidget(self.from_spin, 1, 1)
        s2.addWidget(self.to_lbl, 1, 2)
        s2.addWidget(self.to_spin, 1, 3)
        s2.addWidget(QLabel("Slice thickness:"), 2, 0)
        s2.addWidget(self.thk_spin, 2, 1)
        s2.addWidget(QLabel("Average:"), 3, 0)
        s2.addWidget(self.avg_combo, 3, 1, 1, 3)
        s2.addWidget(self.doi_check, 4, 0, 1, 2)
        s2.addWidget(self.doi_combo, 4, 2, 1, 2)
        pv.addWidget(g2)

        g3 = QGroupBox("3. Interpolation")
        s3 = QGridLayout(g3)
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Inverse distance (IDW)", "Ordinary kriging"])
        self.method_combo.currentIndexChanged.connect(self._method_changed)
        self.spacing_spin = QDoubleSpinBox()
        self.radius_spin = QDoubleSpinBox()
        for sp, val in ((self.spacing_spin, 25), (self.radius_spin, 150)):
            sp.setRange(0.1, 1e6)
            sp.setDecimals(1)
            sp.setValue(val)
            sp.setSuffix(" m")
        self.spacing_spin.setToolTip("Grid node spacing (cell size)")
        self.radius_spin.setToolTip("Search radius; nodes farther than this from any sounding stay blank")
        self.suggest_btn = QPushButton("Suggest")
        self.suggest_btn.setToolTip("Spacing = ½ and radius = 3 × the median distance between neighbouring soundings")
        self.suggest_btn.clicked.connect(self._suggest)
        self.kmax_spin = QSpinBox()
        self.kmax_spin.setRange(1, 200)
        self.kmax_spin.setValue(12)
        self.power_spin = QDoubleSpinBox()
        self.power_spin.setRange(0.5, 6)
        self.power_spin.setSingleStep(0.5)
        self.power_spin.setValue(2.0)
        self.vmodel_combo = QComboBox()
        self.vmodel_combo.addItems(["spherical", "exponential", "gaussian"])
        self.vrange_spin = QDoubleSpinBox()
        self.vrange_spin.setRange(0, 1e6)
        self.vrange_spin.setDecimals(0)
        self.vrange_spin.setSuffix(" m")
        self.vrange_spin.setSpecialValueText("auto")
        self.vrange_spin.setToolTip("Variogram range; 'auto' fits it to the data of each slice")
        s3.addWidget(QLabel("Method:"), 0, 0)
        s3.addWidget(self.method_combo, 0, 1, 1, 3)
        s3.addWidget(QLabel("Node spacing:"), 1, 0)
        s3.addWidget(self.spacing_spin, 1, 1)
        s3.addWidget(self.suggest_btn, 1, 2, 1, 2)
        s3.addWidget(QLabel("Search radius:"), 2, 0)
        s3.addWidget(self.radius_spin, 2, 1)
        s3.addWidget(QLabel("Max points:"), 2, 2)
        s3.addWidget(self.kmax_spin, 2, 3)
        self.power_lbl = QLabel("IDW power:")
        s3.addWidget(self.power_lbl, 3, 0)
        s3.addWidget(self.power_spin, 3, 1)
        self.vmodel_lbl, self.vrange_lbl = QLabel("Variogram:"), QLabel("Range:")
        s3.addWidget(self.vmodel_lbl, 4, 0)
        s3.addWidget(self.vmodel_combo, 4, 1)
        s3.addWidget(self.vrange_lbl, 4, 2)
        s3.addWidget(self.vrange_spin, 4, 3)
        pv.addWidget(g3)

        g4 = QGroupBox("4. Color scale")
        s4 = QGridLayout(g4)
        self.cmap_combo = QComboBox()
        self.cmap_combo.setEditable(True)
        self.cmap_combo.addItems([c for c in RESISTIVITY_CMAPS if get_colormap(c) is not None])
        self.cmap_combo.setCurrentText('BlueToPink')
        self.rev_check = QCheckBox("Reverse")
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["Logarithmic", "Linear"])
        self.cmin_spin, self.cmax_spin = QDoubleSpinBox(), QDoubleSpinBox()
        for sp, val in ((self.cmin_spin, 1.0), (self.cmax_spin, 1000.0)):
            sp.setRange(0.0, 1e9)
            sp.setDecimals(2)
            sp.setValue(val)
            sp.setSuffix(" Ohm·m")
            sp.setKeyboardTracking(False)
        self.cauto_btn = QPushButton("Auto from data")
        self.cauto_btn.clicked.connect(self._auto_range)
        s4.addWidget(QLabel("Colormap:"), 0, 0)
        s4.addWidget(self.cmap_combo, 0, 1)
        s4.addWidget(self.rev_check, 0, 2)
        s4.addWidget(QLabel("Scale:"), 1, 0)
        s4.addWidget(self.scale_combo, 1, 1, 1, 2)
        s4.addWidget(QLabel("Min:"), 2, 0)
        s4.addWidget(self.cmin_spin, 2, 1, 1, 2)
        s4.addWidget(QLabel("Max:"), 3, 0)
        s4.addWidget(self.cmax_spin, 3, 1, 1, 2)
        s4.addWidget(self.cauto_btn, 4, 1, 1, 2)
        pv.addWidget(g4)
        self._cs_timer = QTimer(self)
        self._cs_timer.setSingleShot(True)
        self._cs_timer.setInterval(150)
        self._cs_timer.timeout.connect(self._render)
        for sig in (self.cmap_combo.currentTextChanged, self.rev_check.toggled,
                    self.scale_combo.currentIndexChanged, self.cmin_spin.valueChanged,
                    self.cmax_spin.valueChanged):
            sig.connect(lambda *_: self._cs_timer.start())

        g5 = QGroupBox("5. Map output")
        s5 = QGridLayout(g5)
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("temporary folder")
        self.out_btn = QPushButton("…")
        self.out_btn.setMaximumWidth(30)
        self.out_btn.clicked.connect(self._pick_out_dir)
        s5.addWidget(QLabel("Save rasters in:"), 0, 0)
        s5.addWidget(self.out_edit, 0, 1)
        s5.addWidget(self.out_btn, 0, 2)
        pv.addWidget(g5)
        pv.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(330)
        split.addWidget(scroll)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)

        self.status_label = QLabel("Tick sounding layers, set the slices and interpolation, "
                                   "then Compute Slices.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.canvas.mpl_connect('motion_notify_event', self._on_move)
        self.canvas.mpl_connect('button_press_event', self._on_click)
        self._method_changed()
        self.param_combo.currentIndexChanged.connect(self._param_changed)
        self.picker.selectionChangedByUser.connect(self._update_param_ui)
        self._update_param_ui()
        self._update_nav()
        self._refresh_polygons()
        self._area_mode_changed()

    # ------------------------------------------------------------------
    def _ref_changed(self, *args):
        if self.ref_combo.currentIndex() == 0:
            self.from_lbl.setText("From:")
            self.to_lbl.setText("To:")
            self.from_spin.setValue(0)
            self.to_spin.setValue(100)
        else:
            self.from_lbl.setText("Top:")
            self.to_lbl.setText("Bottom:")
            df = self._soundings(quiet=True)
            if df is not None:
                top = float(np.nanmax(pd.to_numeric(df['ELEVATION'], errors='coerce')))
                self.from_spin.setValue(round(top / 10) * 10)
                self.to_spin.setValue(round(top / 10) * 10 - 100)

    # ------------------------------------------------------------------
    #  Parameter (resistivity or IP) and its colour scale
    # ------------------------------------------------------------------
    def _update_param_ui(self, *args):
        avail = set()
        for lyr in self.picker.checked_layers():
            try:
                avail.update(available_params(pd.DataFrame(columns=self.registry.sounding_columns(lyr))))
            except Exception:
                pass
        self._params_avail = [p for p in PARAM_ORDER if p in avail] or ['RHO']
        ip = len(self._params_avail) > 1
        self.param_lbl.setVisible(ip)
        self.param_combo.setVisible(ip)
        cur = self.param_combo.currentData() or 'RHO'
        self.param_combo.blockSignals(True)
        self.param_combo.clear()
        for prm in self._params_avail:
            self.param_combo.addItem(PARAMS[prm]['label'], prm)
        k = self.param_combo.findData(cur)
        self.param_combo.setCurrentIndex(k if k >= 0 else 0)
        self.param_combo.blockSignals(False)
        if (self.param_combo.currentData() or 'RHO') != self._cs_param:
            self._param_changed()

    def _param(self):
        return self.param_combo.currentData() or 'RHO'

    def _widget_cs(self):
        return dict(cmap=self.cmap_combo.currentText(), rev=self.rev_check.isChecked(),
                    log=self.scale_combo.currentIndex() == 0,
                    vmin=self.cmin_spin.value(), vmax=self.cmax_spin.value())

    def _default_cs(self, prm, values=None):
        if prm == 'RHO':
            return dict(cmap='BlueToPink', rev=False, log=True, vmin=1.0, vmax=1000.0)
        log = prm != 'C'
        lo, hi = (0.0, 1.0) if prm == 'C' else (1.0, 1000.0)
        v = np.asarray(values if values is not None else [], dtype=float)
        v = v[np.isfinite(v) & ((v > 0) if log else True)]
        if v.size:
            lo, hi = np.percentile(v, [2, 98])
            if hi <= lo:
                hi = lo * 10 if log else lo + 1
        return dict(cmap='BlueToPink', rev=False, log=log, vmin=float(f"{lo:.3g}"), vmax=float(f"{hi:.3g}"))

    def _load_cs(self, prm):
        cs = self._cs.get(prm) or self._default_cs(prm)
        info = PARAMS[prm]
        ws = (self.cmap_combo, self.rev_check, self.scale_combo, self.cmin_spin, self.cmax_spin)
        for w in ws:
            w.blockSignals(True)
        self.cmap_combo.setCurrentText(cs['cmap'])
        self.rev_check.setChecked(cs['rev'])
        self.scale_combo.setCurrentIndex(0 if cs['log'] else 1)
        for sp in (self.cmin_spin, self.cmax_spin):
            sp.setDecimals(info['decimals'])
            sp.setSuffix(f" {info['unit']}" if info['unit'] else "")
        self.cmin_spin.setValue(cs['vmin'])
        self.cmax_spin.setValue(cs['vmax'])
        for w in ws:
            w.blockSignals(False)

    def _param_changed(self, *args):
        new = self._param()
        self._cs[self._cs_param] = self._widget_cs()
        self._cs_param = new
        if new not in self._cs:
            vals = []
            for lyr in self.picker.checked_layers():
                try:
                    vals.append(model_arrays(self.registry.soundings_df(lyr), new)[0].ravel())
                except Exception:
                    pass
            self._cs[new] = self._default_cs(new, np.concatenate(vals) if vals else None)
        self._load_cs(new)
        # conductivity averaging only makes sense for resistivity
        cur = self.avg_combo.currentIndex()
        self.avg_combo.blockSignals(True)
        self.avg_combo.clear()
        items = ["Logarithmic (geometric)", "Arithmetic"] + (["Conductivity"] if new == 'RHO' else [])
        self.avg_combo.addItems(items)
        self.avg_combo.setCurrentIndex(cur if cur < len(items) else 0)
        self.avg_combo.blockSignals(False)
        if self.slices and self.slices[0].get('param', 'RHO') != new:
            self.status_label.setText(f"Parameter set to {PARAMS[new]['label']} – press Compute Slices.")

    def _cs_for(self, prm):
        if prm == self._cs_param:
            return self._widget_cs()
        return self._cs.get(prm) or self._default_cs(prm)

    def _method_changed(self, *args):
        kr = self.method_combo.currentIndex() == 1
        for w in (self.power_lbl, self.power_spin):
            w.setVisible(not kr)
        for w in (self.vmodel_lbl, self.vmodel_combo, self.vrange_lbl, self.vrange_spin):
            w.setVisible(kr)
        self.kmax_spin.setValue(16 if kr else 12)

    # ------------------------------------------------------------------
    #  Area
    # ------------------------------------------------------------------
    def _area_mode_changed(self, *args):
        m = self.area_combo.currentIndex()
        self.draw_btn.setVisible(m == 2)
        for w in (self.poly_combo, self.poly_refresh, self.poly_sel_check):
            w.setVisible(m == 3)
        if self._area_rb is not None:
            self._area_rb.setVisible(m == 2 and self._drawn_rect is not None)
        if m == 2 and self._drawn_rect is None:
            self.area_info.setText("Click 'Draw on map' and drag a rectangle.")
        elif m == 2:
            r = self._drawn_rect
            self.area_info.setText(f"Rectangle {r.width():.0f} × {r.height():.0f} (map units).")
        else:
            self.area_info.setText("")

    def _refresh_polygons(self):
        cur = self.poly_combo.currentData()
        self.poly_combo.clear()
        for lid, lyr in QgsProject.instance().mapLayers().items():
            if isinstance(lyr, QgsVectorLayer) and lyr.geometryType() == Qgis.GeometryType.Polygon:
                self.poly_combo.addItem(lyr.name(), lid)
        if cur:
            k = self.poly_combo.findData(cur)
            if k >= 0:
                self.poly_combo.setCurrentIndex(k)

    def _start_draw(self):
        if self.iface is None:
            return
        from qgis.gui import QgsMapToolExtent
        canvas = self.iface.mapCanvas()
        if self._area_tool is None:
            self._area_tool = QgsMapToolExtent(canvas)
            self._area_tool.extentChanged.connect(self._on_drawn)
        cur = canvas.mapTool()
        if cur is not self._area_tool:
            self._prev_map_tool = cur
        canvas.setMapTool(self._area_tool)
        self.area_info.setText("Drag a rectangle on the map…")

    def _end_draw(self):
        if self.iface is None or self._area_tool is None:
            return
        canvas = self.iface.mapCanvas()
        if canvas.mapTool() is self._area_tool:
            try:
                if self._prev_map_tool is not None:
                    canvas.setMapTool(self._prev_map_tool)
                else:
                    canvas.unsetMapTool(self._area_tool)
            except Exception:
                canvas.unsetMapTool(self._area_tool)

    def _on_drawn(self, rect):
        self._end_draw()
        if rect is None or rect.isEmpty():
            return
        self._drawn_rect = QgsRectangle(rect)
        canvas = self.iface.mapCanvas()
        if self._area_rb is None:
            self._area_rb = QgsRubberBand(canvas, Qgis.GeometryType.Polygon)
            self._area_rb.setColor(QColor(255, 0, 0, 200))
            self._area_rb.setFillColor(QColor(255, 0, 0, 25))
            self._area_rb.setWidth(2)
        self._area_rb.setToGeometry(QgsGeometry.fromRect(self._drawn_rect), None)
        self.area_combo.setCurrentIndex(2)
        self._area_mode_changed()

    def _area(self):
        """Area in the data CRS as (xmin, ymin, xmax, ymax, rings or None); None = whole data."""
        m = self.area_combo.currentIndex()
        if m == 0:
            return None
        proj = QgsProject.instance()
        if m in (1, 2):
            if self.iface is None:
                return None
            canvas = self.iface.mapCanvas()
            rect = canvas.extent() if m == 1 else self._drawn_rect
            if rect is None:
                raise ValueError("Draw a rectangle on the map first.")
            src = canvas.mapSettings().destinationCrs()
            if self.crs is not None and self.crs.isValid() and src != self.crs:
                rect = QgsCoordinateTransform(src, self.crs, proj).transformBoundingBox(rect)
            return (rect.xMinimum(), rect.yMinimum(), rect.xMaximum(), rect.yMaximum(), None)
        lyr = proj.mapLayer(self.poly_combo.currentData() or '')
        if lyr is None:
            raise ValueError("Choose a polygon layer (click ↻ to refresh the list).")
        feats = list(lyr.selectedFeatures()) if self.poly_sel_check.isChecked() else list(lyr.getFeatures())
        if not feats:
            raise ValueError(f"No {'selected ' if self.poly_sel_check.isChecked() else ''}polygons in '{lyr.name()}'.")
        tr = None
        if self.crs is not None and self.crs.isValid() and lyr.crs() != self.crs:
            tr = QgsCoordinateTransform(lyr.crs(), self.crs, proj)
        rings = []
        for f in feats:
            g = QgsGeometry(f.geometry())
            if tr is not None:
                g.transform(tr)
            polys = g.asMultiPolygon() if g.isMultipart() else [g.asPolygon()]
            for poly in polys:
                if poly:
                    arr = [np.array([[p.x(), p.y()] for p in ring]) for ring in poly]
                    rings.append((arr[0], arr[1:]))
        allp = np.vstack([r[0] for r in rings])
        return (allp[:, 0].min(), allp[:, 1].min(), allp[:, 0].max(), allp[:, 1].max(), rings)

    def stop_tools(self):
        self._end_draw()

    def set_map_items_visible(self, visible):
        if self._area_rb is not None:
            self._area_rb.setVisible(visible and self.area_combo.currentIndex() == 2
                                     and self._drawn_rect is not None)

    def cleanup(self):
        self._end_draw()
        if self._area_rb is not None and self.iface is not None:
            try:
                self.iface.mapCanvas().scene().removeItem(self._area_rb)
            except Exception:
                pass
            self._area_rb = None

    def _pick_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Folder for MR rasters")
        if d:
            self.out_edit.setText(d)

    def _soundings(self, quiet=False):
        layers = self.picker.checked_layers()
        if not layers:
            if not quiet:
                QMessageBox.warning(self, "No sounding layers", "Tick at least one sounding layer.")
            return None
        df, self.crs = self.registry.combine_soundings(layers)
        return df

    def _suggest(self):
        df = self._soundings()
        if df is None or len(df) < 2:
            return
        xy = df[['X', 'Y']].to_numpy(dtype=float)
        try:
            from scipy.spatial import cKDTree
            d = cKDTree(xy).query(xy, k=2)[0][:, 1]
        except ImportError:
            dd = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
            np.fill_diagonal(dd, np.inf)
            d = dd.min(axis=1)
        med = float(np.median(d[d > 0])) if (d > 0).any() else 10.0
        self.spacing_spin.setValue(max(round(med / 2, 1), 0.1))
        self.radius_spin.setValue(round(3 * med, 1))
        self.status_label.setText(f"Median distance between neighbouring soundings: {med:.1f} m.")

    def _intervals(self):
        a, b, t = self.from_spin.value(), self.to_spin.value(), self.thk_spin.value()
        if self.ref_combo.currentIndex() == 0:
            lo, hi = min(a, b), max(a, b)
            edges = list(np.arange(lo, hi, t)) + [hi]
            return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1) if edges[i + 1] > edges[i]]
        hi, lo = max(a, b), min(a, b)
        edges = list(np.arange(hi, lo, -t)) + [lo]
        return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1) if edges[i] > edges[i + 1]]

    # ------------------------------------------------------------------
    def compute(self):
        df = self._soundings()
        if df is None:
            return
        intervals = self._intervals()
        if not intervals:
            QMessageBox.warning(self, "Slices", "Check From / To / Slice thickness.")
            return
        if len(intervals) > 200:
            QMessageBox.warning(self, "Slices", f"{len(intervals)} slices – use a larger slice thickness.")
            return
        spacing, radius = self.spacing_spin.value(), self.radius_spin.value()
        try:
            area = self._area()
        except ValueError as exc:
            QMessageBox.warning(self, "Area", str(exc))
            return
        if area is not None:
            x0, y0, x1, y1, rings = area
            near = ((df['X'] >= x0 - radius) & (df['X'] <= x1 + radius) &
                    (df['Y'] >= y0 - radius) & (df['Y'] <= y1 + radius))
            df = df[near].reset_index(drop=True)
            if df.empty:
                QMessageBox.warning(self, "Area", "No soundings in or near the chosen area.")
                return
            xs, ys = make_grid(np.array([x0, x1]), np.array([y0, y1]), spacing, margin=0)
        xy = df[['X', 'Y']].to_numpy(dtype=float)
        if area is None:
            xs, ys = make_grid(xy[:, 0], xy[:, 1], spacing, margin=spacing)
        if len(xs) * len(ys) > 4_000_000:
            QMessageBox.warning(self, "Grid too large",
                                f"{len(xs)} × {len(ys)} nodes – increase the node spacing.")
            return
        ref = 'depth' if self.ref_combo.currentIndex() == 0 else 'elevation'
        avg = ('log', 'arith', 'cond')[self.avg_combo.currentIndex()]
        param = self._param()
        if param != 'RHO' and avg == 'cond':
            avg = 'log'
        doi_col = None
        if self.doi_check.isChecked():
            di = self.doi_combo.currentIndex()
            for c in (PARAMS[param]['doi'][di], PARAMS['RHO']['doi'][di], PARAMS[param]['doi'][0], 'DOI_STANDARD'):
                if c in df.columns:
                    doi_col = c
                    break
        try:
            means = slice_means(df, intervals, ref, avg, doi_col, param)
        except Exception as exc:
            QMessageBox.critical(self, "MR maps", str(exc))
            return
        GX, GY = np.meshgrid(xs, ys)
        nodes = np.column_stack([GX.ravel(), GY.ravel()])
        outside = None
        if area is not None and area[4] is not None:
            outside = ~polygon_mask(area[4], nodes).reshape(GX.shape)
        kriging = self.method_combo.currentIndex() == 1
        model = self.vmodel_combo.currentText()
        fixed = self.vrange_spin.value() or None
        self.slices = []
        notes = []
        log_interp = param != 'C'        # C (0..1) is interpolated linearly
        for k, (a, b) in enumerate(intervals):
            v = means[:, k]
            ok = np.isfinite(v) & ((v > 0) if log_interp else True)
            if ok.sum() == 0:
                Z = np.full(GX.shape, np.nan)
                info = "no data"
            else:
                lv = np.log10(v[ok]) if log_interp else v[ok]
                if kriging and ok.sum() >= 3:
                    params = fit_variogram(xy[ok], lv, model, fixed)
                    z, params = ordinary_kriging(xy[ok], lv, nodes, radius, self.kmax_spin.value(),
                                                 model, params)
                    info = (f"kriging {model}, range {params['range']:.0f} m, "
                            f"nugget {params['nugget']:.3f}, sill {params['sill']:.3f}")
                else:
                    z = idw(xy[ok], lv, nodes, radius, self.power_spin.value(), self.kmax_spin.value())
                    info = f"IDW p={self.power_spin.value():g}"
                Z = (10 ** z if log_interp else z).reshape(GX.shape)
            if outside is not None:
                Z = np.where(outside, np.nan, Z)
            unit = "m depth" if ref == 'depth' else "m a.s.l."
            label = f"{a:g}–{b:g} {unit}"
            self.slices.append(dict(label=label, a=a, b=b, Z=Z, xs=xs, ys=ys, pts=xy[ok], area=area,
                                    vals=v[ok], info=info, n=int(ok.sum()), ref=ref, param=param,
                                    method='Kriging' if kriging else 'IDW'))
            notes.append(info)
        self.spacing = spacing
        self.slice_combo.blockSignals(True)
        self.slice_combo.clear()
        self.slice_combo.addItems([s['label'] for s in self.slices])
        self.slice_combo.blockSignals(False)
        if param != 'RHO' and param not in self._cs:
            self._cs[param] = self._default_cs(param, means.ravel())
            self._load_cs(param)
        self.cur = 0
        self._show_slice(0)
        self.status_label.setText(
            f"{len(self.slices)} slice(s), grid {len(xs)} × {len(ys)} nodes at {spacing:g} m, "
            f"{len(df)} soundings" + (f", {PARAMS[param]['label']}." if param != 'RHO' else "."))

    # ------------------------------------------------------------------
    def _step(self, d):
        if self.slices:
            self.slice_combo.setCurrentIndex(max(0, min(len(self.slices) - 1, self.cur + d)))

    def _update_nav(self):
        n = len(self.slices)
        self.prev_btn.setEnabled(n > 0 and self.cur > 0)
        self.next_btn.setEnabled(n > 0 and self.cur < n - 1)
        for w in (self.add_btn, self.add_all_btn, self.slice_combo):
            w.setEnabled(n > 0)

    def _show_slice(self, i):
        if not self.slices or i < 0:
            return
        self.cur = int(i)
        self._render()
        self._update_nav()
        s = self.slices[self.cur]
        self.status_label.setText(f"Slice {s['label']}: {s['n']} soundings, {s['info']}.")

    def _slice_param(self):
        return self.slices[self.cur].get('param', 'RHO') if self.slices else self._param()

    def _cmap_norm(self, prm=None):
        cs = self._cs_for(prm or self._slice_param())
        cmap = get_colormap(cs['cmap'], cs['rev']) or get_colormap('BlueToPink')
        vmin, vmax = cs['vmin'], cs['vmax']
        if cs['log']:
            vmin = vmin if vmin > 0 else 1e-9
            if vmin >= vmax:
                vmax = vmin * 10
            return cmap, matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
        if vmin >= vmax:
            vmax = vmin + 1
        return cmap, matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    @staticmethod
    def _names(prm):
        """(title word, colourbar label, unit text) of a parameter."""
        if prm == 'RHO':
            return "Mean resistivity", 'Resistivity (Ohm·m)', 'Ohm·m'
        info = PARAMS[prm]
        return f"Mean {info['label']}", info['cb'], (info['unit'] or info['label'])

    def _render(self):
        self.figure.clf()
        self.ax = self.ax_depth = None
        if not self.slices:
            self.canvas.draw_idle()
            return
        s = self.slices[self.cur]
        prm = s.get('param', 'RHO')
        cmap, norm = self._cmap_norm(prm)
        title, cb_label, unit = self._names(prm)
        fig = self.figure
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 3], height_ratios=[1.5, 1],
                              left=0.08, right=0.90, bottom=0.08, top=0.93, wspace=0.28, hspace=0.42)
        ax = fig.add_subplot(gs[:, 1])
        sp = s['xs'][1] - s['xs'][0] if len(s['xs']) > 1 else 1.0
        ext = (s['xs'][0] - sp / 2, s['xs'][-1] + sp / 2, s['ys'][0] - sp / 2, s['ys'][-1] + sp / 2)
        im = ax.imshow(np.ma.masked_invalid(s['Z']), origin='lower', extent=ext, cmap=cmap,
                       norm=norm, interpolation='nearest')
        if self.show_pts_check.isChecked() and len(s['pts']):
            ax.scatter(s['pts'][:, 0], s['pts'][:, 1], s=4, c='black', linewidths=0)
        area = s.get('area')
        if area is not None:
            if area[4] is not None:
                for ext_ring, holes in area[4]:
                    ax.plot(ext_ring[:, 0], ext_ring[:, 1], color='black', linewidth=0.8)
            else:
                x0, y0, x1, y1 = area[:4]
                ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color='black', linewidth=0.8)
        ax.set_aspect('equal')
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.ticklabel_format(useOffset=False, style='plain')
        ax.tick_params(labelsize=8)
        ax.text(0.0, 1.01, f"{title} {s['label']}  ({s['method']})",
                transform=ax.transAxes, ha='left', va='bottom', fontsize=10)
        fig.colorbar(im, ax=ax, label=cb_label, shrink=0.85, pad=0.02, fraction=0.04)
        self.ax = ax

        # --- left top: mean resistivity vs depth for all slices (click = choose slice)
        axd = fig.add_subplot(gs[0, 0])
        mids, med, p10, p90 = [], [], [], []
        for sl in self.slices:
            v = sl['vals']
            mids.append((sl['a'] + sl['b']) / 2.0)
            if len(v):
                med.append(np.median(v)); p10.append(np.percentile(v, 10)); p90.append(np.percentile(v, 90))
            else:
                med.append(np.nan); p10.append(np.nan); p90.append(np.nan)
        mids, med, p10, p90 = map(np.asarray, (mids, med, p10, p90))
        axd.fill_betweenx(mids, p10, p90, color='0.85', linewidth=0)
        axd.plot(med, mids, color='black', linewidth=1)
        okm = np.isfinite(med)
        axd.scatter(med[okm], mids[okm], c=cmap(norm(med[okm])), s=22, edgecolors='black',
                    linewidths=0.4, zorder=3)
        axd.axhspan(min(s['a'], s['b']), max(s['a'], s['b']), color='#ffcc00', alpha=0.45, zorder=1)
        if isinstance(norm, matplotlib.colors.LogNorm):
            axd.set_xscale('log')
        depth = s['ref'] == 'depth'
        axd.set_ylabel("Depth (m)" if depth else "Elevation (m)", fontsize=8)
        axd.set_xlabel(f"{unit} (median, 10–90 %)", fontsize=8)
        if depth:
            axd.invert_yaxis()
        axd.tick_params(labelsize=7)
        axd.set_title("All slices – click to choose", fontsize=8)
        self.ax_depth = axd

        # --- left bottom: histogram of the soundings in this slice
        axh = fig.add_subplot(gs[1, 0])
        v = s['vals'][np.isfinite(s['vals']) &
                      ((s['vals'] > 0) if isinstance(norm, matplotlib.colors.LogNorm) else True)]
        if v.size:
            if isinstance(norm, matplotlib.colors.LogNorm):
                bins = np.logspace(np.log10(v.min()), np.log10(v.max()) + 1e-9, 16)
                axh.set_xscale('log')
            else:
                bins = np.linspace(v.min(), v.max() + 1e-9, 16)
            n, edges, patches = axh.hist(v, bins=bins, edgecolor='white', linewidth=0.4)
            for p_, lo, hi in zip(patches, edges[:-1], edges[1:]):
                p_.set_facecolor(cmap(norm(np.sqrt(lo * hi) if lo > 0 else (lo + hi) / 2)))
        axh.set_xlabel(unit, fontsize=8)
        axh.set_ylabel("Soundings", fontsize=8)
        axh.tick_params(labelsize=7)
        axh.set_title(f"This slice: {s['n']} soundings", fontsize=8)
        for a in (axd, axh):
            if a.get_xscale() == 'log':
                a.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=6))
                a.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:.3g}"))
                a.xaxis.set_minor_formatter(NullFormatter())
        self.canvas.draw_idle()

    def _auto_range(self):
        vals = np.concatenate([s['Z'][np.isfinite(s['Z'])].ravel() for s in self.slices]) \
            if self.slices else np.array([])
        if self.scale_combo.currentIndex() == 0:
            vals = vals[vals > 0]
        if not vals.size:
            return
        lo, hi = np.percentile(vals, [2, 98])
        for sp, v in ((self.cmin_spin, lo), (self.cmax_spin, hi)):
            sp.blockSignals(True)
            sp.setValue(float(f"{v:.3g}"))
            sp.blockSignals(False)
        self._render()

    def _on_click(self, ev):
        if not self.slices or ev.inaxes is None or ev.inaxes is not getattr(self, 'ax_depth', None):
            return
        for k, sl in enumerate(self.slices):
            if min(sl['a'], sl['b']) <= ev.ydata <= max(sl['a'], sl['b']):
                self.slice_combo.setCurrentIndex(k)
                return

    def _on_move(self, ev):
        if not self.slices or ev.inaxes is None or ev.xdata is None or ev.inaxes is not self.ax:
            return
        s = self.slices[self.cur]
        sp = s['xs'][1] - s['xs'][0] if len(s['xs']) > 1 else 1.0
        ix = int(round((ev.xdata - s['xs'][0]) / sp))
        iy = int(round((ev.ydata - s['ys'][0]) / sp))
        val = ""
        if 0 <= iy < s['Z'].shape[0] and 0 <= ix < s['Z'].shape[1] and np.isfinite(s['Z'][iy, ix]):
            prm = s.get('param', 'RHO')
            val = f"{s['Z'][iy, ix]:.4g} {self._names(prm)[2]}"
        sym = 'ρ' if s.get('param', 'RHO') == 'RHO' else PARAMS[s['param']]['label']
        self.cursor_label.setText(f"X: {ev.xdata:.1f}   Y: {ev.ydata:.1f}   {sym}: {val}")

    # ------------------------------------------------------------------
    #  QGIS map layers
    # ------------------------------------------------------------------
    def _style_layer(self, lyr, prm='RHO'):
        from qgis.core import QgsColorRampShader, QgsRasterShader, QgsSingleBandPseudoColorRenderer
        cmap, norm = self._cmap_norm(prm)
        vmin, vmax = norm.vmin, norm.vmax
        n = 32
        if isinstance(norm, matplotlib.colors.LogNorm):
            values = np.logspace(np.log10(vmin), np.log10(vmax), n)
        else:
            values = np.linspace(vmin, vmax, n)
        items = []
        for v in values:
            r, g, b, a = cmap(norm(v))
            items.append(QgsColorRampShader.ColorRampItem(float(v), QColor(int(r * 255), int(g * 255), int(b * 255)),
                                                          f"{v:.3g}"))
        shader = QgsColorRampShader()
        shader.setColorRampType(qgis_enum(Qgis, 'ShaderInterpolationMethod', 'Linear')
                                if hasattr(Qgis, 'ShaderInterpolationMethod')
                                else qgis_enum(QgsColorRampShader, 'Type', 'Interpolated'))
        shader.setColorRampItemList(items)
        shader.setClip(False)
        rs = QgsRasterShader()
        rs.setRasterShaderFunction(shader)
        ren = QgsSingleBandPseudoColorRenderer(lyr.dataProvider(), 1, rs)
        ren.setClassificationMin(vmin)
        ren.setClassificationMax(vmax)
        lyr.setRenderer(ren)

    def _add_to_map(self, all_slices=False):
        if not self.slices:
            return
        out = self.out_edit.text().strip()
        if not out:
            if not getattr(self, '_tmpdir', None):
                self._tmpdir = tempfile.mkdtemp(prefix="tv_mr_")
            out = self._tmpdir
        os.makedirs(out, exist_ok=True)
        proj = QgsProject.instance()
        root = proj.layerTreeRoot()
        group = root.findGroup("TEM-Viewer MR maps") or root.insertGroup(0, "TEM-Viewer MR maps")
        wkt = self.crs.toWkt() if self.crs is not None else proj.crs().toWkt()
        todo = self.slices if all_slices else [self.slices[self.cur]]
        added = []
        for s in todo:
            prm = s.get('param', 'RHO')
            pre = 'MR' if prm == 'RHO' else f"M{prm}"
            tag = re.sub(r'[^0-9A-Za-z.-]+', '_', s['label'])
            path = os.path.join(out, f"{pre}_{tag}_{s['method']}.tif")
            path = write_raster(path, s['Z'], s['xs'], s['ys'], self.spacing, wkt)
            name = (f"MR {s['label']} ({s['method']})" if prm == 'RHO' else
                    f"Mean {PARAMS[prm]['label']} {s['label']} ({s['method']})")
            lyr = QgsRasterLayer(path, name)
            if not lyr.isValid():
                QMessageBox.warning(self, "Raster", f"Could not load {path}")
                continue
            if self.crs is not None and self.crs.isValid():
                lyr.setCrs(self.crs)
            self._style_layer(lyr, prm)
            proj.addMapLayer(lyr, False)
            group.insertLayer(0, lyr)
            added.append(lyr.name())
        if added:
            self.status_label.setText(f"Added {len(added)} raster layer(s) to the map (saved in {out}).")
