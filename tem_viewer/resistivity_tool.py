# -*- coding: utf-8 -*-
"""
TEM-Viewer - Profile Tool

Plots TEM/AEM 1D resistivity models (.xyz model files) as a
resistivity section along a line picked on the map, through soundings selected
on the map, or along a line handed over by the TEM-Viewer.
"""
import os
import re
import json
import traceback
import numpy as np
import pandas as pd
from .tv_compat import (FigureCanvasQTAgg as FigureCanvas,
                             NavigationToolbar2QT as NavigationToolbar, qgis_enum)
import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.collections
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

from qgis.PyQt.QtCore import Qt, QVariant, QTimer
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QMessageBox, QGroupBox, QComboBox, QCheckBox,
    QSplitter, QWidget, QDoubleSpinBox, QScrollArea, QRadioButton, QInputDialog
)
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry, QgsPointXY,
    QgsField, QgsPalLayerSettings, QgsVectorLayerSimpleLabeling,
    QgsMarkerSymbol, QgsSingleSymbolRenderer, QgsWkbTypes,
    QgsCoordinateReferenceSystem, QgsCoordinateTransform, Qgis
)
from qgis.gui import QgsRubberBand, QgsVertexMarker, QgsMapTool
import math

from .tv_data import DataRegistry, LayerPicker
from .borehole_section import (boreholes_near_line, draw_boreholes, lith_at, draw_sounding_frames,
                               draw_water_levels, WL_HANDLE)
from . import water_level as wlv
from .tv_compat import FIELD_DOUBLE, FIELD_INT, FIELD_STRING

PROFILE_KIND = 'temviewer_profiles'     # custom property of saved profile-line layers
TOPO_COLOR = '#6b3e1f'   # topography line on the section
BAR_REF_PX = 1000.0      # 'px' bar widths are pixels of a section this wide (≈ A4-landscape report)
PROFILE_FIELDS = [("name", FIELD_STRING), ("length_m", FIELD_DOUBLE), ("start", FIELD_STRING),
                  ("end", FIELD_STRING), ("search_w", FIELD_DOUBLE), ("n_snd", FIELD_INT),
                  ("layer_ids", FIELD_STRING), ("layers", FIELD_STRING), ("uid", FIELD_STRING),
                  ("report_data", FIELD_STRING), ("settings", FIELD_STRING)]



class SortableTableItem(QTableWidgetItem):
    """
    QTableWidgetItem that sorts numerically when the cell text looks like a number,
    otherwise falls back to normal string comparison.
    """

    def __lt__(self, other):
        try:
            return float(self.text()) < float(other.text())
        except ValueError:
            return self.text() < other.text()


def read_xyz_file(file_path, dummy_values=(9999, -9999)):
    """
    Read an .xyz model file.

    Supports:
      * plain files whose first line is the column header, and
      * exports with a '/'-commented metadata block, where the
        header is the last comment line before the data (e.g. '/ LINE_NO X Y ...').
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    if not lines:
        raise ValueError("No non-empty lines found in .xyz file.")

    def is_comment(ln):
        return ln.lstrip().startswith('/')

    # --- locate first data line ------------------------------------------
    first_data = next((i for i, ln in enumerate(lines) if not is_comment(ln)), None)
    if first_data is None:
        raise ValueError("No data lines found in .xyz file.")
    n_data_cols = len(lines[first_data].split())

    # --- locate header ----------------------------------------------------
    headers = None
    if first_data == 0:
        # Old style: header is the first (non-commented) line
        headers = lines[0].split()
        data_start = 1
    else:
        # commented-header style: walk back through the comment block and take the
        # nearest comment line whose token count matches the data width
        data_start = first_data
        for ln in reversed(lines[:first_data]):
            tokens = ln.lstrip().lstrip('/').split()
            if len(tokens) == n_data_cols and not all(
                    re.match(r'^[-+0-9.eE]+$', t) for t in tokens):
                headers = tokens
                break
    if not headers:
        raise ValueError(
            "Could not find a column header line matching the data width "
            f"({n_data_cols} columns).")

    # --- de-duplicate header names ---------------------------------------
    seen, unique_headers = {}, []
    for h in headers:
        if h in seen:
            seen[h] += 1
            unique_headers.append(f"{h}_{seen[h]}")
        else:
            seen[h] = 0
            unique_headers.append(h)

    # --- data -------------------------------------------------------------
    data_lines = [ln.split() for ln in lines[data_start:] if not is_comment(ln)]
    bad = [i for i, row in enumerate(data_lines) if len(row) != len(unique_headers)]
    if bad:
        raise ValueError(
            f"{len(bad)} data row(s) have a column count different from the header "
            f"({len(unique_headers)}). First bad row index: {bad[0]}")

    df = pd.DataFrame(data_lines, columns=unique_headers)
    for col in df.columns:
        if col in ('DATE', 'TIME'):
            continue                      # keep as text
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # dummy value (/DUMMY 9999) -> NaN
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace(list(dummy_values), np.nan)
    return df


RESISTIVITY_CMAPS = [
    'BlueToPink', 'Spectral', 'jet', 'turbo', 'rainbow', 'nipy_spectral', 'RdYlBu',
    'coolwarm', 'seismic', 'viridis', 'plasma', 'inferno', 'magma',
    'cividis', 'terrain', 'gist_earth', 'YlGnBu', 'gray',
]


# Blue-to-pink resistivity scale ("TEM_BlueToPink"): low = blue ... high = pink
BLUEPINK_STOPS = [
    (0.0000000, (0, 0, 190)), (0.0666667, (0, 75, 220)), (0.1333333, (0, 150, 235)),
    (0.2000000, (0, 200, 255)), (0.2666667, (80, 240, 255)), (0.3333333, (30, 210, 0)),
    (0.4000000, (180, 255, 30)), (0.4666667, (255, 255, 0)), (0.5333333, (255, 195, 0)),
    (0.6000000, (255, 115, 0)), (0.6666667, (255, 0, 0)), (0.7333333, (255, 0, 120)),
    (0.8000000, (140, 40, 180)), (0.8666667, (165, 70, 220)), (0.9333333, (195, 130, 240)),
    (1.0000000, (230, 155, 240)),
]


def bluepink_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        'BlueToPink', [(p, tuple(c / 255.0 for c in rgb)) for p, rgb in BLUEPINK_STOPS])


def register_bluepink_ramp():
    """Add the blue-to-pink scale to the QGIS colour ramps (Style Manager) as
    'TEM_BlueToPink', so it can also be used for layer styling."""
    try:
        from qgis.PyQt.QtGui import QColor
        from qgis.core import QgsGradientColorRamp, QgsGradientStop, QgsStyle
        style = QgsStyle.defaultStyle()
        if "TEM_BlueToPink" in style.colorRampNames():
            return
        stops = [QgsGradientStop(p, QColor(*rgb)) for p, rgb in BLUEPINK_STOPS[1:-1]]
        ramp = QgsGradientColorRamp(QColor(*BLUEPINK_STOPS[0][1]), QColor(*BLUEPINK_STOPS[-1][1]),
                                    False, stops)
        style.addColorRamp("TEM_BlueToPink", ramp)
    except Exception:
        pass


def get_colormap(name, reverse=False):
    """Return a private copy of a matplotlib colormap (None if the name is unknown).

    Works with both old (cm.get_cmap) and new (matplotlib.colormaps) APIs.
    """
    name = (name or '').strip()
    if name.lower() in ('bluetopink', 'bluetopink_r'):
        cmap = bluepink_cmap()
        if reverse != name.lower().endswith('_r'):
            cmap = cmap.reversed()
        cmap.set_bad(color='white', alpha=0)
        return cmap
    if reverse:
        name = name[:-2] if name.endswith('_r') else name + '_r'
    try:
        registry = getattr(matplotlib, 'colormaps', None)
        cmap = registry[name] if registry is not None else matplotlib.cm.get_cmap(name)
    except (KeyError, ValueError):
        return None
    cmap = cmap.copy()
    cmap.set_bad(color='white', alpha=0)
    return cmap


def read_xyz_crs(file_path):
    """Return the CRS declared in a '/COORDINATE SYSTEM' header block, or None."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for _ in range(500):
                ln = f.readline()
                if not ln or not ln.lstrip().startswith('/'):
                    break
                m = re.search(r'epsg\s*[:=]\s*(\d+)', ln, re.I)
                if m:
                    crs = QgsCoordinateReferenceSystem(f"EPSG:{m.group(1)}")
                    return crs if crs.isValid() else None
    except Exception:
        pass
    return None


def get_sounding_layers(df, record_id, elev_col='ELEVATION',
                        rho_prefix='RHO_', dep_top_prefix='DEP_TOP_', thk_prefix='THK_',
                        coord='elevation', id_col='RECORD', doi_col='DOI_STANDARD'):
    """Model layers of one sounding. Layers below the DOI in `doi_col` are cut
    off; doi_col=None returns the full model."""
    sounding = df[df[id_col] == record_id]
    if sounding.empty:
        return pd.DataFrame()
    sounding = sounding.iloc[0]
    surface_elev = sounding[elev_col]
    doi_val = sounding.get(doi_col) if doi_col else None
    doi_limit = None
    if pd.notna(doi_val) and float(doi_val) > 0:
        doi_limit = float(doi_val) if coord == 'depth' else (float(surface_elev) - float(doi_val))

    def sort_numbered(cols):
        return sorted(cols, key=lambda x: int(re.findall(r'\d+$', x)[0]))
    rho_cols = sort_numbered([c for c in df.columns if re.match(rf'{rho_prefix}\d+$', c)])
    dep_cols = sort_numbered([c for c in df.columns if re.match(rf'{dep_top_prefix}\d+$', c)])
    thk_cols = sort_numbered([c for c in df.columns if re.match(rf'{thk_prefix}\d+$', c)])
    n_layers = len(rho_cols)
    if n_layers == 0:
        return pd.DataFrame()
    layers = []
    prev_bottom = None
    for i in range(n_layers):
        dep_col = dep_cols[i] if i < len(dep_cols) else None
        thk_col = thk_cols[i] if i < len(thk_cols) else None
        if dep_col and pd.notna(sounding.get(dep_col)):
            top = sounding[dep_col] if coord == 'depth' else (surface_elev - sounding[dep_col])
        else:
            top = (0.0 if coord == 'depth' else surface_elev) if i == 0 else prev_bottom
        if thk_col and pd.notna(sounding.get(thk_col)):
            bottom = top + sounding[thk_col] if coord == 'depth' else top - sounding[thk_col]
        elif (i + 1 < len(dep_cols)) and pd.notna(sounding.get(dep_cols[i + 1])):
            v = sounding[dep_cols[i + 1]]
            bottom = v if coord == 'depth' else (surface_elev - v)
        else:
            bottom = top + 1.0 if coord == 'depth' else top - 1.0
        # ---- DOI clip -------------------------------------------------
        if doi_limit is not None:
            if coord == 'depth':
                if top > doi_limit:
                    continue
                if bottom > doi_limit:
                    bottom = doi_limit
            else:
                if top < doi_limit:
                    continue
                if bottom < doi_limit:
                    bottom = doi_limit
        # ----------------------------------------------------------------
        val = sounding[rho_cols[i]]
        layers.append({
            'Top': float(top),
            'Bottom': float(bottom),
            'Resistivity': float(val) if pd.notna(val) else None
        })
        prev_bottom = bottom
    return pd.DataFrame(layers)


def select_ids(layer, ids):
    """layer.selectByIds that works on QGIS 3 and 4 (list or set of feature ids)."""
    ids = [int(i) for i in ids]
    try:
        layer.selectByIds(ids)
    except TypeError:
        layer.selectByIds(set(ids))


def _log(msg):
    try:
        from qgis.core import QgsMessageLog
        QgsMessageLog.logMessage(msg, "TEM-Viewer", Qgis.MessageLevel.Info)
    except Exception:
        pass


# Model parameters that can be shown in the profile (IP inversions add PHI, TPHI, C)
PARAMS = {
    'RHO':  dict(label='RES', cb='Resistivity (Ohm·m)', unit='Ohm·m', export='Resistivity',
                 doi=('DOI_STANDARD', 'DOI_CONSERVATIVE'), decimals=2),
    'PHI':  dict(label='PHI', cb='Phase φ (mrad)', unit='mrad', export='Phi',
                 doi=('PHI_DOI_STA', 'PHI_DOI_CON'), decimals=2),
    'TPHI': dict(label='TPHI (τ)', cb='τφ (s)', unit='s', export='TauPhi',
                 doi=('TPHI_DOI_STA', 'TPHI_DOI_CON'), decimals=7),
    'C':    dict(label='C', cb='C (–)', unit='', export='C',
                 doi=('C_DOI_STA', 'C_DOI_CON'), decimals=3),
    'M0':   dict(label='M0', cb='Chargeability m0 (mV/V)', unit='mV/V', export='Chargeability',
                 doi=('M0_DOI_STA', 'M0_DOI_CON'), decimals=2),
    'TAU':  dict(label='τ', cb='τ (s)', unit='s', export='Tau',
                 doi=('TAU_DOI_STA', 'TAU_DOI_CON'), decimals=7),
}
PARAM_ORDER = ['RHO', 'PHI', 'TPHI', 'C', 'M0', 'TAU']

# numbered columns that are not model parameters (geometry, uncertainties, DOI, data ...)
_NOT_PARAM = re.compile(r'^(DEP_TOP|DEP_BOT|THK|DEP|DEPTH|TOP|BOT|Z|ELEV.*|.*STD.*|.*DOI.*|.*ERR.*|'
                        r'.*UNC.*|.*VAR.*|.*SENS.*|.*DATA.*|.*GATE.*|.*RES(DATA|TOTAL).*|.*MISFIT.*|_.*)$')


def register_params(columns):
    """Any other model parameter in the file (a numbered column family with one value per
    layer, e.g. XYZ_1..XYZ_N) becomes a parameter that can be plotted and mapped."""
    L = len(_numbered_cols(columns, 'RHO_'))
    if not L:
        return []
    fams = {}
    for c in columns:
        m = re.match(r'^([A-Z][A-Z0-9_]*?)_(\d+)$', str(c))
        if m:
            fams.setdefault(m.group(1), set()).add(int(m.group(2)))
    new = []
    for prefix, nums in fams.items():
        if prefix in PARAMS or _NOT_PARAM.match(prefix) or len(nums) != L:
            continue
        PARAMS[prefix] = dict(label=prefix, cb=prefix, unit='', export=prefix,
                              doi=(f'{prefix}_DOI_STA', f'{prefix}_DOI_CON'), decimals=4)
        PARAM_ORDER.append(prefix)
        new.append(prefix)
    return new


def find_doi_column(columns, names, kind_idx):
    """DOI column for a parameter: exact name, then case-insensitive, then a fuzzy
    match (e.g. 'DOI_Standard', 'doi_std', 'DOI-Conservative'). None if not found."""
    cols = [str(c) for c in columns]
    up = {c.upper(): c for c in cols}
    want = names[kind_idx]
    if want.upper() in up:
        return up[want.upper()]
    if names[0].upper().startswith('DOI_'):          # plain resistivity DOI only
        keys = ('STAND', 'STD', 'STA') if kind_idx == 0 else ('CONS', 'CON')
        for c in cols:
            u = re.sub(r'[^A-Z]', '', c.upper())
            if u.startswith('DOI') and any(k in u for k in keys):
                return c
    return None


def _numbered_cols(cols, prefix):
    out = [c for c in cols if re.match(re.escape(prefix) + r'\d+$', str(c))]
    return sorted(out, key=lambda c: int(re.findall(r'\d+$', c)[0]))


def available_params(df):
    """Model parameters present in the data (RHO always first)."""
    register_params(list(df.columns))
    return [p for p in PARAM_ORDER if _numbered_cols(df.columns, p + '_')]


def layer_geometry(df):
    """Top and bottom depth (m below surface) of every layer, n_soundings x n_layers."""
    L = len(_numbered_cols(df.columns, 'RHO_'))
    num = lambda cols: df[cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float, copy=True)
    top_c, bot_c, thk_c = (_numbered_cols(df.columns, 'DEP_TOP_'), _numbered_cols(df.columns, 'DEP_BOT_'),
                           _numbered_cols(df.columns, 'THK_'))
    n = len(df)
    if len(top_c) >= L:
        top = num(top_c[:L])
    elif thk_c:
        thk = num(thk_c)
        top = np.zeros((n, L))
        top[:, 1:] = np.cumsum(thk[:, :L - 1], axis=1)
    else:
        raise ValueError("No DEP_TOP_i or THK_i columns to define the layer depths.")
    bot = np.full((n, L), np.nan)
    if bot_c:
        b = num(bot_c)[:, :L]
        bot[:, :b.shape[1]] = b
    bot[:, :-1] = np.where(np.isfinite(bot[:, :-1]), bot[:, :-1], top[:, 1:])
    if len(thk_c) >= L:
        bot[:, -1] = np.where(np.isfinite(bot[:, -1]), bot[:, -1], top[:, -1] + num([thk_c[L - 1]])[:, 0])
    last = np.where(np.isfinite(bot[:, -1]), bot[:, -1], top[:, -1] * 1.25 + 10.0)
    bot[:, -1] = last
    return top, bot


def layer_values(df, param):
    cols = _numbered_cols(df.columns, param + '_')
    return df[cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float, copy=True)


class LineDrawTool(QgsMapTool):
    """Draw a polyline on the map: left-click adds vertices, right-click or Enter
    finishes, Backspace removes the last vertex, Esc cancels."""

    def __init__(self, canvas, on_done, on_cancel=None):
        super().__init__(canvas)
        self.on_done = on_done
        self.on_cancel = on_cancel
        self.pts = []
        self.rb = QgsRubberBand(canvas, Qgis.GeometryType.Line)
        self.rb.setColor(QColor(255, 0, 0, 180))
        self.rb.setWidth(2)
        self.rb.setLineStyle(Qt.PenStyle.DashLine)

    def _redraw(self, floating=None):
        self.rb.reset(Qgis.GeometryType.Line)
        for p in self.pts + ([floating] if floating is not None else []):
            self.rb.addPoint(p)

    def start(self):
        self.pts = []
        self._redraw()

    def canvasMoveEvent(self, e):
        if self.pts:
            self._redraw(e.mapPoint())

    def canvasReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.RightButton:
            self._finish()
        elif e.button() == Qt.MouseButton.LeftButton:
            self.pts.append(QgsPointXY(e.mapPoint()))
            self._redraw()

    def keyPressEvent(self, e):
        k = e.key()
        if k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._finish()
        elif k == Qt.Key.Key_Backspace and self.pts:
            self.pts.pop()
            self._redraw()
        elif k == Qt.Key.Key_Escape:
            self.pts = []
            self._redraw()
            if self.on_cancel:
                self.on_cancel()

    def _finish(self):
        pts = list(self.pts)
        self.pts = []
        self._redraw()
        if len(pts) >= 2:
            self.on_done(pts)
        elif self.on_cancel:
            self.on_cancel()

    def deactivate(self):
        self.pts = []
        self._redraw()
        super().deactivate()


class ResistivityProfileWidget(QWidget):
    """Resistivity profile (section) through 1D TEM/AEM models.

    Used as a tab in TEM-Viewer - TEM-Viewer (embedded=True, soundings
    come from the TEM data tab) and as a stand-alone window -
    Resistivity Profile Tool window (embedded=False, own .xyz loading).
    """

    def __init__(self, parent=None, iface=None, embedded=False, registry=None):
        super().__init__(parent)
        self.iface = iface
        self.embedded = embedded
        self.registry = registry if registry is not None else DataRegistry(self)
        self._sel_conns = []       # sounding layers whose selection we follow
        self._single = False       # profile of one sounding only
        self.models_df = None
        self.profile_points = []
        self.map_tool = None
        self.rb_line = None
        self.ax = None
        self.cursor_hline = None
        self.cursor_vline = None
        self._last_plot_data = None
        self.data_crs = None       # CRS of the combined sounding coordinates
        self._color_artist = None  # PatchCollection / QuadMesh coloured by resistivity
        self._cbar = None          # current colorbar
        self._rho_data = None      # resistivity values in the current plot
        self.ax2 = None            # secondary axis for the data misfit
        self._axes = []            # one axis per plotted parameter (top to bottom)
        self._panels = {}          # param -> {'ax', 'artist', 'cbar', 'data'}
        self._cs = {}              # param -> colour-scale settings
        self._cs_param = 'RHO'     # parameter shown in the Color scale box
        self._params_avail = ['RHO']
        self._cursor_artists = []
        self._nav = None           # Previous/Next stepping along the line (option a)
        self._nav_len = 0.0        # window length kept while stepping (0 = one sounding)
        self._navigating = False
        self._draw_tool = None     # polyline drawing tool (option b)
        self._drawn_line = None    # drawn line in the data CRS
        self._prev_map_tool = None
        self._line_geom = None     # profile line (data CRS) of the current plot
        self._has_plot = False
        self.rb_arrow = None       # arrow head at the end of the profile line
        self.start_marker = None   # circle at the start of the profile line
        self.pos_marker = None     # map marker following the plot cursor
        self._setup_ui()
        if self.iface:
            self.rb_line = QgsRubberBand(self.iface.mapCanvas(), Qgis.GeometryType.Line)
            self.rb_line.setColor(Qt.GlobalColor.red)
            self.rb_line.setWidth(2)
            canvas = self.iface.mapCanvas()
            self.rb_arrow = QgsRubberBand(canvas, Qgis.GeometryType.Polygon)
            self.rb_arrow.setColor(Qt.GlobalColor.red)
            self.rb_arrow.setFillColor(Qt.GlobalColor.red)
            self.start_marker = QgsVertexMarker(canvas)
            self.start_marker.setIconType(qgis_enum(QgsVertexMarker, 'IconType', 'ICON_CIRCLE'))
            self.start_marker.setColor(Qt.GlobalColor.red)
            self.start_marker.setFillColor(Qt.GlobalColor.white)
            self.start_marker.setIconSize(12)
            self.start_marker.setPenWidth(3)
            self.start_marker.hide()
            self.pos_marker = QgsVertexMarker(canvas)
            self.pos_marker.setIconType(qgis_enum(QgsVertexMarker, 'IconType', 'ICON_CROSS'))
            self.pos_marker.setColor(Qt.GlobalColor.blue)
            self.pos_marker.setIconSize(18)
            self.pos_marker.setPenWidth(3)
            self.pos_marker.hide()
            canvas.extentsChanged.connect(self._update_direction_marks)
        try:
            from .borehole_section import color_events
            color_events().changed.connect(self._lith_colors_changed)
        except Exception:
            pass

    # -----------------------------------------------------------------
    #  UI – plot on the left, all settings in one scrollable panel on the right
    # -----------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)

        # ---- Left: plot ---------------------------------------------
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(2)
        self.figure = Figure(figsize=(7, 5))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(400, 300)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.cursor_label = QLabel("X: \\nY: ")
        self.cursor_label.setStyleSheet(
            "font-family: monospace; font-size: 10px; border: 1px solid #cccccc; padding: 3px;")
        self.cursor_label.setMinimumHeight(36)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas, 1)
        plot_layout.addWidget(self.cursor_label)

        bottom_h = QHBoxLayout()
        self.plot_btn = QPushButton("Plot Profile")
        self.plot_btn.setToolTip("Plot through the selected soundings (a) or along the "
                                 "line from the layer (b), as chosen on the right")
        self.plot_btn.setStyleSheet("font-weight: bold;")
        self.plot_btn.clicked.connect(lambda: self._plot_profile())
        bottom_h.addWidget(self.plot_btn)
        self.clear_btn = QPushButton("Clear profile")
        self.clear_btn.setToolTip("Remove the plotted profile and the profile line from the map")
        self.clear_btn.clicked.connect(self.clear_profile)
        bottom_h.addWidget(self.clear_btn)
        self.yaxis_combo = QComboBox()
        self.yaxis_combo.addItems(["Elevation", "Depth"])
        self.yaxis_combo.setToolTip("Plot against elevation (m) or depth below surface (m)")
        bottom_h.addWidget(self.yaxis_combo)
        self.reverse_btn = QPushButton("⇄ Reverse")
        self.reverse_btn.setCheckable(True)
        self.reverse_btn.setToolTip("Flip the profile direction (e.g. W → E becomes E → W)")
        bottom_h.addWidget(self.reverse_btn)
        self.reset_view_btn = QPushButton("Reset view")
        self.reset_view_btn.clicked.connect(self._reset_view)
        bottom_h.addWidget(self.reset_view_btn)
        self.show_cursor_check = QCheckBox("Cursor on map")
        self.show_cursor_check.setChecked(True)
        self.show_cursor_check.setToolTip("Show the cursor on the plot and its position "
                                          "along the profile on the map")
        bottom_h.addWidget(self.show_cursor_check)
        bottom_h.addStretch()
        bottom_h.addWidget(QLabel("Save as:"))
        self.save_format_combo = QComboBox()
        self.save_format_combo.addItems([
            "Graph - PNG", "Graph - PDF", "Graph + location map - PDF", "Graph - SVG",
            "Data - CSV", "Data - Excel"])
        bottom_h.addWidget(self.save_format_combo)
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self._save_profile)
        bottom_h.addWidget(self.save_btn)
        self.save_layer_btn = QPushButton("Save Line as Layer")
        self.save_layer_btn.setToolTip("Add the plotted profile line (with its direction, search width and "
                                       "sounding layers) to the 'TEM-Viewer profiles' map layer, e.g. for the "
                                       "Geological Model tab")
        self.save_layer_btn.clicked.connect(self._save_profile_layer)
        bottom_h.addWidget(self.save_layer_btn)
        plot_layout.addLayout(bottom_h)
        splitter.addWidget(plot_container)

        # ---- Right: settings ----------------------------------------
        panel = QWidget()
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(0, 0, 4, 0)
        pv.setSpacing(6)

        # 1. Sounding layers (one or several)
        snd_group = QGroupBox("1. Sounding layers")
        sg = QVBoxLayout(snd_group)
        self.xyz_edit = QLineEdit()
        self.xyz_btn = QPushButton("Browse…")
        self.xyz_btn.clicked.connect(self._pick_xyz)
        self.load_btn = QPushButton("Load")
        self.load_btn.clicked.connect(self._load_xyz)
        if not self.embedded:
            fh = QHBoxLayout()
            fh.addWidget(self.xyz_edit, 1)
            fh.addWidget(self.xyz_btn)
            fh.addWidget(self.load_btn)
            sg.addLayout(fh)
            self.xyz_edit.setPlaceholderText(".xyz file(s) – each becomes a sounding layer")
        hint = QLabel("Tick the layers to take soundings from"
                      + (" (import .xyz files on the Soundings tab):" if self.embedded else ":"))
        hint.setWordWrap(True)
        sg.addWidget(hint)
        self.snd_picker = LayerPicker(self.registry, 'soundings', self, max_height=80)
        self.snd_picker.selectionChangedByUser.connect(self._follow_selections)
        sg.addWidget(self.snd_picker)
        pv.addWidget(snd_group)

        # 2. Profile line
        line_group = QGroupBox("2. Profile line")
        lv = QVBoxLayout(line_group)
        lv.setSpacing(4)

        self.src_draw_radio = QRadioButton("a) Along a line drawn on the map")
        self.src_draw_radio.setChecked(True)
        lv.addWidget(self.src_draw_radio)
        dr_h = QHBoxLayout()
        dr_h.setContentsMargins(18, 0, 0, 0)
        self.draw_btn = QPushButton("Draw line on map")
        self.draw_btn.setToolTip("Left-click to add points, right-click (or Enter) to finish;\n"
                                 "Backspace removes the last point, Esc cancels.\n"
                                 "The profile is plotted right away from the soundings within the "
                                 "search width.")
        self.draw_btn.clicked.connect(self._start_draw)
        dr_h.addWidget(self.draw_btn)
        dr_h.addStretch()
        lv.addLayout(dr_h)

        self.src_sel_radio = QRadioButton("b) Through soundings selected on the map")
        lv.addWidget(self.src_sel_radio)
        ss_h = QHBoxLayout()
        ss_h.setContentsMargins(18, 0, 0, 0)
        self.select_snd_btn = QPushButton("Select soundings on map")
        self.select_snd_btn.setToolTip(
            "Shows the soundings on the map and activates the rectangle-select tool.\n"
            "Shift+drag adds to the selection, Ctrl+drag removes.")
        self.select_snd_btn.clicked.connect(self._activate_sounding_selection)
        ss_h.addWidget(self.select_snd_btn)
        lv.addLayout(ss_h)
        self.sel_count_label = QLabel("0 soundings selected.")
        self.sel_count_label.setContentsMargins(18, 0, 0, 0)
        lv.addWidget(self.sel_count_label)
        nav_h = QHBoxLayout()
        nav_h.setContentsMargins(18, 0, 0, 0)
        self.prev_btn = QPushButton("◀ Previous")
        self.next_btn = QPushButton("Next ▶")
        for b, step in ((self.prev_btn, -1), (self.next_btn, 1)):
            b.setEnabled(False)
            b.setToolTip("Step along the line by one sounding, or by a window as long as "
                         "the plotted selection, and plot it")
            b.clicked.connect(lambda _=False, st=step: self._nav_step(st))
        self.nav_label = QLabel("")
        self.nav_label.setWordWrap(True)
        nav_h.addWidget(self.prev_btn)
        nav_h.addWidget(self.next_btn)
        lv.addLayout(nav_h)
        self.nav_label.setContentsMargins(18, 0, 0, 0)
        lv.addWidget(self.nav_label)

        self.src_line_radio = QRadioButton("c) Along a line from a layer")
        self.src_line_radio.setToolTip("Uses the selected line of the layer, otherwise its first line")
        lv.addWidget(self.src_line_radio)
        ll_h = QHBoxLayout()
        ll_h.setContentsMargins(18, 0, 0, 0)
        self.line_layer_combo = QComboBox()
        self.line_layer_combo.setMinimumWidth(120)
        self.line_layer_combo.currentIndexChanged.connect(
            lambda *_: self.src_line_radio.setChecked(True))
        self.line_refresh_btn = QPushButton("↻")
        self.line_refresh_btn.setMaximumWidth(28)
        self.line_refresh_btn.setToolTip("Refresh the list of line layers")
        self.line_refresh_btn.clicked.connect(self._refresh_line_layers)
        ll_h.addWidget(self.line_layer_combo, 1)
        ll_h.addWidget(self.line_refresh_btn)
        lv.addLayout(ll_h)

        self.src_saved_radio = QRadioButton("d) A saved profile")
        self.src_saved_radio.setToolTip("A profile stored with 'Save Line as Layer' (group 'TEM-Viewer profiles')")
        lv.addWidget(self.src_saved_radio)
        sp_h = QHBoxLayout()
        sp_h.setContentsMargins(18, 0, 0, 0)
        self.saved_combo = QComboBox()
        self.saved_combo.setMinimumWidth(120)
        self.saved_combo.currentIndexChanged.connect(self._on_saved_combo_changed)
        self.saved_load_btn = QPushButton("Load")
        self.saved_load_btn.setToolTip("Tick the profile's sounding layers, restore its plot settings "
                                       "(search width, y-axis, colour scales, ...) and plot it")
        self.saved_load_btn.clicked.connect(self._load_saved_profile)
        sp_h.addWidget(self.saved_combo, 1)
        sp_h.addWidget(self.saved_load_btn)
        lv.addLayout(sp_h)
        se_h = QHBoxLayout()
        se_h.setContentsMargins(18, 0, 0, 0)
        self.saved_edit_btn = QPushButton("Edit line")
        self.saved_edit_btn.setCheckable(True)
        self.saved_edit_btn.setToolTip("Move, add or delete vertices of the saved line on the map with the "
                                       "QGIS vertex tool; the profile is re-plotted after each change.\n"
                                       "Click again to finish (keep or discard the changes).")
        self.saved_edit_btn.toggled.connect(self._toggle_edit_saved)
        self.saved_update_btn = QPushButton("Update saved profile")
        self.saved_update_btn.setToolTip("Overwrite the loaded saved profile with the current plot: line, "
                                         "sounding layers and plot settings")
        self.saved_update_btn.clicked.connect(self._update_saved_profile)
        se_h.addWidget(self.saved_edit_btn)
        se_h.addWidget(self.saved_update_btn)
        lv.addLayout(se_h)
        self.saved_info_label = QLabel("")
        self.saved_info_label.setWordWrap(True)
        self.saved_info_label.setContentsMargins(18, 0, 0, 0)
        lv.addWidget(self.saved_info_label)
        self._loaded_profile = None      # (layer id, feature id) of the loaded saved profile
        self._edit_lyr_id = None         # id of the saved-profile layer being edited on the map
        self._loading = False            # restoring settings: no automatic re-plots
        self._geom_timer = QTimer(self)
        self._geom_timer.setSingleShot(True)
        self._geom_timer.setInterval(300)
        self._geom_timer.timeout.connect(self._on_saved_geom_edited)
        self.registry.changed.connect(self._refresh_saved_profiles)

        self.direction_label = QLabel("Direction: –")
        self.direction_label.setToolTip("Start → end of the profile. Use ⇄ Reverse under the plot to flip it.")
        lv.addWidget(self.direction_label)
        pv.addWidget(line_group)

        # 3. Plot settings
        opt_group = QGroupBox("3. Plot settings")
        og = QGridLayout(opt_group)
        og.setSpacing(4)
        self.tolerance_edit = QLineEdit("10")
        self.tolerance_edit.setMaximumWidth(70)
        self.tolerance_edit.setToolTip("For options a and c: max distance of soundings from the line (m)")
        og.addWidget(QLabel("Search width (m):"), 1, 0)
        og.addWidget(self.tolerance_edit, 1, 1)
        self.barwidth_edit = QLineEdit("10")
        self.barwidth_edit.setMaximumWidth(70)
        self.barwidth_edit.editingFinished.connect(self._replot_if_plotted)
        self.barunit_combo = QComboBox()
        self.barunit_combo.addItem("m", 'm')
        self.barunit_combo.addItem("px", 'px')
        self.barunit_combo.setToolTip(
            "m = bar width in metres along the profile (wide bars on short profiles, thin on long ones).\n"
            "px = bar width in pixels on a section 1000 px wide (about an A4-landscape report page):\n"
            "the bars take the same share of the section on every profile, so they look equally wide\n"
            "in reports whatever the profile length.")
        self._bw_vals = {'m': "10", 'px': "8"}
        self._bw_unit = 'm'
        self.barunit_combo.currentIndexChanged.connect(self._bar_unit_changed)
        bw_w = QWidget()
        bw_h = QHBoxLayout(bw_w)
        bw_h.setContentsMargins(0, 0, 0, 0)
        bw_h.addWidget(self.barwidth_edit)
        bw_h.addWidget(self.barunit_combo)
        bw_h.addStretch()
        og.addWidget(QLabel("Bar width:"), 2, 0)
        og.addWidget(bw_w, 2, 1)
        self.interp_check = QCheckBox("Interpolate between soundings")
        og.addWidget(self.interp_check, 3, 0)
        self.frames_check = QCheckBox("Sounding frames")
        self.frames_check.setToolTip("Outline each sounding (bar width) and its layers, "
                                     "so you can see where the soundings are")
        og.addWidget(self.frames_check, 3, 1)

        # IP inversions: choose which model parameters to plot (stacked panels)
        self.param_row = QWidget()
        pr_h = QHBoxLayout(self.param_row)
        pr_h.setContentsMargins(0, 0, 0, 0)
        pr_h.addWidget(QLabel("Plot:"))
        self.param_checks = {}
        for prm in PARAM_ORDER:
            cb = QCheckBox(PARAMS[prm]['label'])
            cb.setChecked(prm == 'RHO')
            cb.setToolTip("Tick several to stack them (top to bottom)")
            cb.toggled.connect(self._replot_if_plotted)
            pr_h.addWidget(cb)
            self.param_checks[prm] = cb
        pr_h.addStretch()
        self.param_row.setVisible(False)
        og.addWidget(self.param_row, 9, 0, 1, 2)

        self.doi_mask_check = QCheckBox("Mask below DOI")
        self.doi_mask_check.setChecked(True)
        self.doi_mask_check.setToolTip("Shade the model below the depth of investigation")
        self.doi_col_combo = QComboBox()
        self.doi_col_combo.addItems(["Standard", "Conservative"])
        self.doi_col_combo.setToolTip("DOI_STANDARD or DOI_CONSERVATIVE from the .xyz file")
        self.doi_opacity_spin = QDoubleSpinBox()
        self.doi_opacity_spin.setRange(0, 100)
        self.doi_opacity_spin.setDecimals(0)
        self.doi_opacity_spin.setSingleStep(10)
        self.doi_opacity_spin.setValue(100)
        self.doi_opacity_spin.setSuffix(" %")
        self.doi_opacity_spin.setKeyboardTracking(False)
        self.doi_opacity_spin.setToolTip("How strongly the model below the DOI is masked: "
                                         "0 % = fully visible, 100 % = hidden")
        og.addWidget(self.doi_mask_check, 4, 0, 1, 2)
        og.addWidget(QLabel("DOI:"), 5, 0)
        og.addWidget(self.doi_col_combo, 5, 1)
        og.addWidget(QLabel("Mask strength:"), 6, 0)
        og.addWidget(self.doi_opacity_spin, 6, 1)

        self.misfit_check = QCheckBox("Show data misfit (right axis)")
        self.misfit_col_combo = QComboBox()
        self.misfit_col_combo.addItems(["RESDATA", "RESTOTAL"])
        self.misfit_col_combo.setToolTip("RESDATA = data residual, RESTOTAL = total residual")
        og.addWidget(self.misfit_check, 7, 0, 1, 2)
        og.addWidget(QLabel("Misfit:"), 8, 0)
        og.addWidget(self.misfit_col_combo, 8, 1)
        self.misfit_max_spin = QDoubleSpinBox()
        self.misfit_max_spin.setRange(0.1, 1e6)
        self.misfit_max_spin.setDecimals(1)
        self.misfit_max_spin.setValue(10)
        self.misfit_max_spin.setKeyboardTracking(False)
        self.misfit_max_spin.setToolTip("The right axis runs from 0 to this value")
        og.addWidget(QLabel("Misfit axis 0 to:"), 9, 0)
        og.addWidget(self.misfit_max_spin, 9, 1)
        self.misfit_max_spin.valueChanged.connect(self._replot_if_plotted)
        self.front_edit = QLineEdit()
        self.front_edit.setMaximumWidth(70)
        self.front_edit.setPlaceholderText("record no.")
        self.front_edit.setToolTip("Where soundings overlap (bar width larger than their spacing), draw "
                                   "the sounding with this record number in front. Empty = normal order.")
        og.addWidget(QLabel("Sounding in front:"), 13, 0)
        og.addWidget(self.front_edit, 13, 1)
        self.front_edit.editingFinished.connect(self._replot_if_plotted)

        # boreholes near the line, drawn on top of the section
        self.bh_check = QCheckBox("Boreholes within")
        self.bh_check.setToolTip("Draw the boreholes of the ticked borehole layers that lie within this "
                                 "distance of the profile line (projected onto it)")
        self.bh_dist_spin = QDoubleSpinBox()
        self.bh_dist_spin.setRange(0, 1e6)
        self.bh_dist_spin.setDecimals(0)
        self.bh_dist_spin.setValue(100)
        self.bh_dist_spin.setSuffix(" m")
        self.bh_dist_spin.setKeyboardTracking(False)
        self.bh_picker = LayerPicker(self.registry, 'boreholes', self, max_height=55)
        self.bh_picker.setVisible(False)
        og.addWidget(self.bh_check, 10, 0)
        og.addWidget(self.bh_dist_spin, 10, 1)
        og.addWidget(self.bh_picker, 11, 0, 1, 2)
        self.bh_width_spin = QDoubleSpinBox()
        self.bh_width_spin.setRange(0, 1e5)
        self.bh_width_spin.setDecimals(0)
        self.bh_width_spin.setValue(0)
        self.bh_width_spin.setSuffix(" m")
        self.bh_width_spin.setSpecialValueText("auto")
        self.bh_width_spin.setKeyboardTracking(False)
        self.bh_width_spin.setToolTip("Width of the borehole columns; 'auto' = 0.8 × bar width "
                                      "(at least 1.2 % of the profile)")
        self.bh_width_lbl = QLabel("Borehole width:")
        og.addWidget(self.bh_width_lbl, 12, 0)
        og.addWidget(self.bh_width_spin, 12, 1)
        for w_ in (self.bh_width_lbl, self.bh_width_spin):
            w_.setVisible(False)
            self.bh_check.toggled.connect(w_.setVisible)
        self.bh_width_spin.valueChanged.connect(lambda *_: self.bh_check.isChecked() and self._replot_if_plotted())
        # water level in the boreholes, from a column of the borehole data chosen by the user
        self.wl_check = QCheckBox("Water level from:")
        self.wl_check.setToolTip("Draw the water level in each borehole. The value is taken from the chosen "
                                 "column, in the row whose name cell matches the borehole name.")
        self.wl_combo = QComboBox()
        self.wl_combo.setMinimumWidth(120)
        self.wl_combo.setToolTip("Column with the water level (all columns of the ticked borehole layers "
                                 "and of the sheets of their Excel file / GeoPackage)")
        self.wl_mode_combo = QComboBox()
        self.wl_mode_combo.addItem("depth below surface (m)", 'depth')
        self.wl_mode_combo.addItem("elevation (m)", 'elev')
        self.wl_mode_combo.setToolTip("How the values in the column are given")
        self.wl_info = QLabel("")
        self.wl_info.setWordWrap(True)
        self.wl_info.setStyleSheet("color: #555;")
        og.addWidget(self.wl_check, 14, 0)
        og.addWidget(self.wl_combo, 14, 1)
        self.wl_mode_lbl = QLabel("Values are:")
        og.addWidget(self.wl_mode_lbl, 15, 0)
        og.addWidget(self.wl_mode_combo, 15, 1)
        og.addWidget(self.wl_info, 16, 0, 1, 2)
        self._wl_cache = {}
        for w_ in (self.wl_check, self.wl_combo, self.wl_mode_lbl, self.wl_mode_combo, self.wl_info):
            w_.setVisible(False)
            self.bh_check.toggled.connect(w_.setVisible)
        self.bh_check.toggled.connect(lambda on: on and self.wl_check.isChecked() and self._refresh_wl_columns())
        self.wl_check.toggled.connect(lambda on: (on and self._refresh_wl_columns(), self._replot_if_plotted()))
        self.wl_combo.currentIndexChanged.connect(lambda *_: self.wl_check.isChecked() and self._replot_if_plotted())
        self.wl_mode_combo.currentIndexChanged.connect(
            lambda *_: self.wl_check.isChecked() and self._replot_if_plotted())
        # topography along the profile, sampled from a raster (DEM)
        self.topo_check = QCheckBox("Topography from:")
        self.topo_check.setToolTip("Draw the terrain surface along the profile, sampled from a raster layer "
                                   "(Elevation y-axis only).\nUse an elevation model (DEM GeoTIFF): satellite "
                                   "images and hillshades hold colours / shading, not heights.")
        self.topo_combo = QComboBox()
        self.topo_combo.setMinimumWidth(120)
        self.topo_info = QLabel("")
        self.topo_info.setWordWrap(True)
        self.topo_info.setStyleSheet("color: #555;")
        og.addWidget(self.topo_check, 17, 0)
        og.addWidget(self.topo_combo, 17, 1)
        og.addWidget(self.topo_info, 18, 0, 1, 2)
        self._topo_cache = {}
        self.topo_check.toggled.connect(self._replot_if_plotted)
        self.topo_combo.currentIndexChanged.connect(self._topo_layer_changed)
        self.registry.changed.connect(self._refresh_topo_layers)
        self._refresh_topo_layers()
        self.bh_check.toggled.connect(self.bh_picker.setVisible)
        self.bh_check.toggled.connect(self._replot_if_plotted)
        self.bh_dist_spin.valueChanged.connect(lambda *_: self.bh_check.isChecked() and self._replot_if_plotted())
        self.bh_picker.selectionChangedByUser.connect(
            lambda: self.bh_check.isChecked() and self.wl_check.isChecked() and self._refresh_wl_columns())
        self.bh_picker.selectionChangedByUser.connect(
            lambda: self.bh_check.isChecked() and self._replot_if_plotted())
        pv.addWidget(opt_group)

        # 4. Colour scale – changes are applied to the current plot immediately
        cs_group = QGroupBox("4. Color scale")
        cs_grid = QGridLayout(cs_group)
        cs_grid.setSpacing(4)
        self.cs_param_label = QLabel("Parameter:")
        self.cs_param_combo = QComboBox()
        self.cs_param_combo.setToolTip("Each parameter keeps its own colour scale")
        self.cs_param_combo.currentIndexChanged.connect(self._on_cs_param_changed)
        cs_grid.addWidget(self.cs_param_label, 5, 0)
        cs_grid.addWidget(self.cs_param_combo, 5, 1, 1, 2)
        self.cs_param_label.setVisible(False)
        self.cs_param_combo.setVisible(False)
        self.cmap_combo = QComboBox()
        self.cmap_combo.setEditable(True)   # any matplotlib colormap name can be typed
        self.cmap_combo.addItems([c for c in RESISTIVITY_CMAPS if get_colormap(c) is not None])
        self.cmap_combo.setCurrentText('BlueToPink')
        self._last_cmap_name = 'Spectral'
        self.cmap_combo.setToolTip("Pick a colormap or type any matplotlib colormap name.")
        self.cmap_reverse_check = QCheckBox("Reverse")
        self.cmap_reverse_check.setChecked(True)   # Spectral reversed = original default
        cs_grid.addWidget(QLabel("Colormap:"), 0, 0)
        cs_grid.addWidget(self.cmap_combo, 0, 1)
        cs_grid.addWidget(self.cmap_reverse_check, 0, 2)
        self.cscale_combo = QComboBox()
        self.cscale_combo.addItems(["Logarithmic", "Linear"])
        cs_grid.addWidget(QLabel("Scale:"), 1, 0)
        cs_grid.addWidget(self.cscale_combo, 1, 1, 1, 2)
        self.cmin_spin = QDoubleSpinBox()
        self.cmax_spin = QDoubleSpinBox()
        for sp, val in ((self.cmin_spin, 1.0), (self.cmax_spin, 1000.0)):
            sp.setDecimals(2)
            sp.setRange(0.0, 1e9)
            sp.setValue(val)
            sp.setSuffix(" Ohm·m")
            sp.setKeyboardTracking(False)   # update when typing is finished
        cs_grid.addWidget(QLabel("Min:"), 2, 0)
        cs_grid.addWidget(self.cmin_spin, 2, 1, 1, 2)
        cs_grid.addWidget(QLabel("Max:"), 3, 0)
        cs_grid.addWidget(self.cmax_spin, 3, 1, 1, 2)
        self.cauto_btn = QPushButton("Auto from data")
        self.cauto_btn.setToolTip("Set min/max to the 2nd–98th percentile of the plotted values.")
        self.cauto_btn.clicked.connect(self._auto_color_range)
        cs_grid.addWidget(self.cauto_btn, 4, 1, 1, 2)
        self._color_timer = QTimer(self)
        self._color_timer.setSingleShot(True)
        self._color_timer.setInterval(150)
        self._color_timer.timeout.connect(self._apply_color_scale)
        self.cmap_combo.currentTextChanged.connect(lambda *_: self._color_timer.start())
        self.cmap_reverse_check.toggled.connect(lambda *_: self._color_timer.start())
        self.cscale_combo.currentIndexChanged.connect(self._on_scale_type_changed)
        self.cmin_spin.valueChanged.connect(lambda *_: self._color_timer.start())
        self.cmax_spin.valueChanged.connect(lambda *_: self._color_timer.start())
        pv.addWidget(cs_group)
        pv.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(330)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        self.status_label = QLabel(
            "Load or tick sounding layers, choose a profile line (a, b or c), then Plot Profile."
            if not self.embedded else
            "Tick sounding layers, choose a profile line (a, b or c), then Plot Profile.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('axes_leave_event', self._on_mouse_leave)
        self._refresh_line_layers()
        self._refresh_saved_profiles()
        # re-plot automatically when these change
        self.yaxis_combo.currentIndexChanged.connect(self._replot_if_plotted)
        self.reverse_btn.toggled.connect(self._replot_if_plotted)
        self.interp_check.toggled.connect(self._replot_if_plotted)
        self.frames_check.toggled.connect(self._replot_if_plotted)
        for w in (self.doi_mask_check, self.misfit_check):
            w.toggled.connect(self._replot_if_plotted)
        for w in (self.doi_col_combo, self.misfit_col_combo):
            w.currentIndexChanged.connect(self._replot_if_plotted)
        self.doi_opacity_spin.valueChanged.connect(self._replot_if_plotted)
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_name_changed)

    # -----------------------------------------------------------------
    #  Soundings handed over by the main window
    # -----------------------------------------------------------------

    def _refresh_line_layers(self):
        current = self.line_layer_combo.currentData()
        self.line_layer_combo.blockSignals(True)
        self.line_layer_combo.clear()
        for lyr_id, lyr in QgsProject.instance().mapLayers().items():
            if isinstance(lyr, QgsVectorLayer) and lyr.geometryType() == Qgis.GeometryType.Line:
                self.line_layer_combo.addItem(lyr.name(), lyr_id)
        if current:
            idx = self.line_layer_combo.findData(current)
            if idx >= 0:
                self.line_layer_combo.setCurrentIndex(idx)
        self.line_layer_combo.blockSignals(False)
    # -----------------------------------------------------------------
    #  Universal Plot Profile: (a) selected soundings or (b) line layer
    # -----------------------------------------------------------------

    def _gather_soundings(self):
        layers = self.snd_picker.checked_layers()
        if not layers:
            QMessageBox.warning(self, "No sounding layers",
                                "Tick at least one sounding layer under 1. Sounding layers.")
            return False
        try:
            self.models_df, self.data_crs = self.registry.combine_soundings(layers)
        except Exception as exc:
            QMessageBox.critical(self, "Sounding data", str(exc))
            return False
        return True

    def _plot_profile(self):
        if not self._gather_soundings():
            return
        df = self.models_df
        only_ids = None
        self._single = False
        if self.src_sel_radio.isChecked():
            picked = []
            for lyr in self.snd_picker.checked_layers():
                fids = set(lyr.selectedFeatureIds())
                if fids:
                    picked.append(df[(df['_LAYER'] == lyr.id()) & df['_FID'].isin(fids)])
            sel = pd.concat(picked) if picked else df.iloc[0:0]
            if sel.empty:
                QMessageBox.information(
                    self, "Select soundings",
                    "Select one or more soundings on the map (in the ticked layers), "
                    "or choose option c (line from a layer).")
                return
            only_ids = set(sel['RECORD'].astype(float))
            pts = sel[['X', 'Y']].to_numpy(dtype=float)
            if len(pts) == 1:
                # one sounding: a short dummy line so it plots as a single column
                self._single = True
                x, y = pts[0]
                points = [QgsPointXY(x - 0.5, y), QgsPointXY(x + 0.5, y)]
                src_txt = f"sounding {int(sel['_SRC_RECORD'].iloc[0])} of {sel['_SOURCE'].iloc[0]}"
            else:
                centred = pts - pts.mean(axis=0)          # order along the main trend
                _, _, vt = np.linalg.svd(centred, full_matrices=False)
                order = np.argsort(centred @ vt[0])
                points = [QgsPointXY(float(pts[i, 0]), float(pts[i, 1])) for i in order]
                src_txt = f"{len(pts)} selected soundings"
        elif self.src_saved_radio.isChecked():
            points, src_txt = self._line_from_saved()
            if points is None:
                return
        elif self.src_draw_radio.isChecked():
            if not self._drawn_line or len(self._drawn_line) < 2:
                QMessageBox.information(self, "Draw a line",
                                        "Click 'Draw line on map', click points on the map and "
                                        "right-click to finish.")
                return
            points = list(self._drawn_line)
            src_txt = f"drawn line, {len(points)} points"
        else:
            points, src_txt = self._line_from_layer()
            if points is None:
                return
        if self.src_saved_radio.isChecked():
            # a saved profile keeps its stored direction; ⇄ Reverse flips it
            if self.reverse_btn.isChecked():
                points = list(reversed(points))
        elif not self._single:
            points = self._orient(points)
        self.profile_points = points
        self._refresh_rubberband()
        self._draw_profile(only_ids=only_ids)
        if self.src_sel_radio.isChecked() and self._has_plot:
            self._setup_nav(sel, points)
        else:
            self._nav = None
            self._update_nav_ui()
        if self._has_plot:
            if not self.src_sel_radio.isChecked():
                self.prev_btn.setEnabled(False)
                self.next_btn.setEnabled(False)
                self.nav_label.setText("")
            if self._single:
                self.direction_label.setText(f"Single sounding ({src_txt})")
            else:
                st, en = self._direction_labels()
                self.direction_label.setText(f"Direction: {st} → {en}  ({src_txt})")

    # -----------------------------------------------------------------
    #  Selection on the map
    # -----------------------------------------------------------------
    def _follow_selections(self):
        for lyr in self._sel_conns:
            try:
                lyr.selectionChanged.disconnect(self._update_sel_count)
            except Exception:
                pass
        self._sel_conns = self.snd_picker.checked_layers()
        for lyr in self._sel_conns:
            lyr.selectionChanged.connect(self._update_sel_count)
        self._update_sel_count()
        self._update_param_ui()

    def _update_sel_count(self, *args):
        n = 0
        for lyr in self._sel_conns:
            try:
                n += lyr.selectedFeatureCount()
            except RuntimeError:
                pass
        self.sel_count_label.setText(f"{n} sounding{'s' if n != 1 else ''} selected"
                                     f" in the ticked layer{'s' if len(self._sel_conns) != 1 else ''}.")

    def _activate_sounding_selection(self):
        if self.iface is None:
            return
        layers = self.snd_picker.checked_layers()
        if not layers:
            QMessageBox.warning(self, "No sounding layers", "Tick at least one sounding layer first.")
            return
        self.src_sel_radio.setChecked(True)
        self._end_draw()
        active = self.iface.activeLayer()
        if active not in layers:
            self.iface.setActiveLayer(layers[0])
        self.iface.actionSelectRectangle().trigger()
        extra = (" To select in another ticked layer, make it the active layer in the Layers panel."
                 if len(layers) > 1 else "")
        self.status_label.setText(
            "Drag on the map to select soundings (Shift = add, Ctrl = remove), then click "
            "Plot Profile. One sounding is enough." + extra)

    # -----------------------------------------------------------------
    #  Stand-alone: load .xyz files as sounding layers
    # -----------------------------------------------------------------
    def _pick_xyz(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select .xyz file(s)", "", "XYZ files (*.xyz);;All files (*)")
        if paths:
            self.xyz_edit.setText("; ".join(paths))

    def _load_xyz(self):
        paths = [p.strip() for p in self.xyz_edit.text().split(";") if p.strip()]
        loaded = []
        for path in paths:
            if not os.path.isfile(path):
                QMessageBox.warning(self, "Missing file", f"File not found:\n{path}")
                continue
            try:
                df = read_xyz_file(path)
                crs = read_xyz_crs(path) or QgsProject.instance().crs()
                name = f"Soundings - {os.path.splitext(os.path.basename(path))[0]}"
                lyr = self.registry.add_soundings(df, name, crs, path)
                loaded.append(f"{lyr.name()} ({lyr.featureCount()})")
            except Exception as exc:
                QMessageBox.critical(self, "Error", f"{os.path.basename(path)}:\n{exc}")
        if loaded:
            self.status_label.setText("Loaded: " + ", ".join(loaded))

    # -----------------------------------------------------------------
    #  Previous / Next along the line (option a)
    # -----------------------------------------------------------------
    def _setup_nav(self, sel, points):
        df = self.models_df
        track = df[df['_LAYER'].isin(set(sel['_LAYER']))]
        line_col = next((c for c in ('LINE_NO', 'LINE', 'LINENO') if c in df.columns), None)
        if line_col is not None:
            keys = set(zip(sel['_LAYER'], sel[line_col]))
            track = track[[k in keys for k in zip(track['_LAYER'], track[line_col])]]
        # acquisition order (layer, record) and distance along that track
        track = track.sort_values(['_LAYER', '_SRC_RECORD'], kind='stable')
        xy = track[['X', 'Y']].to_numpy(dtype=float)
        if len(xy) < 2:
            self._nav = None
            self._update_nav_ui()
            return
        steps = np.hypot(*np.diff(xy, axis=0).T)
        pos = np.concatenate([[0.0], np.cumsum(steps)])
        # "Next" must head towards the plotted end of the profile
        if not self._single:
            u = np.array([points[-1].x() - points[0].x(), points[-1].y() - points[0].y()])
        else:
            u = xy[-1] - xy[0]
            if (abs(u[0]) >= abs(u[1]) and u[0] < 0) or (abs(u[1]) > abs(u[0]) and u[1] < 0):
                u = -u
            if self.reverse_btn.isChecked():
                u = -u
        if float(np.dot(xy[-1] - xy[0], u)) < 0:
            pos = pos.max() - pos
        key = list(zip(track['_LAYER'], track['_FID']))
        sel_keys = set(zip(sel['_LAYER'], sel['_FID']))
        sel_pos = pos[[k in sel_keys for k in key]]
        win = (float(sel_pos.min()), float(sel_pos.max()))
        if not self._navigating:
            self._nav_len = win[1] - win[0] if not self._single else 0.0
        self._nav = {'pos': pos, 'key': key, 'win': win, 'n_track': len(key),
                     'line': (", ".join(str(v) for v in sorted(set(sel[line_col])))
                              if line_col is not None else None)}
        self._update_nav_ui()

    def _update_nav_ui(self):
        nav = self._nav
        if not nav:
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self.nav_label.setText("")
            return
        pos, (lo, hi) = nav['pos'], nav['win']
        self.prev_btn.setEnabled(bool((pos < lo - 1e-6).any()))
        self.next_btn.setEnabled(bool((pos > hi + 1e-6).any()))
        inside = int(((pos >= lo - 1e-6) & (pos <= hi + 1e-6)).sum())
        step = "1 sounding" if self._nav_len == 0 else f"{self._nav_len:.0f} m"
        line = f" on line {nav['line']}" if nav['line'] else ""
        self.nav_label.setText(f"Step: {step}{line} · showing {inside} of {nav['n_track']} soundings")

    def _nav_step(self, step):
        nav = self._nav
        if not nav:
            return
        pos, (lo, hi), L, eps = nav['pos'], nav['win'], self._nav_len, 1e-6
        if step > 0:
            ahead = pos[pos > hi + eps]
            if not ahead.size:
                self.status_label.setText("End of the line reached.")
                return
            a = ahead.min()
            pick = (pos >= a - eps) & (pos <= a + L + eps)
        else:
            behind = pos[pos < lo - eps]
            if not behind.size:
                self.status_label.setText("Start of the line reached.")
                return
            b = behind.max()
            pick = (pos <= b + eps) & (pos >= b - L - eps)
        if L == 0:                               # exactly one sounding
            target = pos[pick]
            near = target.min() if step > 0 else target.max()
            pick = np.abs(pos - near) <= eps
            first = int(np.argmax(pick))
            pick = np.zeros_like(pick)
            pick[first] = True
        by_layer = {}
        for (lid, fid), p in zip(nav['key'], pick):
            if p:
                by_layer.setdefault(lid, []).append(int(fid))
        for lyr in self.snd_picker.checked_layers():
            select_ids(lyr, by_layer.get(lyr.id(), []))
        self._navigating = True
        try:
            self._plot_profile()
        finally:
            self._navigating = False
        self._pan_to_profile()

    # -----------------------------------------------------------------
    #  Option b: draw a line on the map
    # -----------------------------------------------------------------
    def _start_draw(self):
        if self.iface is None:
            return
        if not self.snd_picker.checked_layers():
            QMessageBox.warning(self, "No sounding layers", "Tick at least one sounding layer first.")
            return
        canvas = self.iface.mapCanvas()
        if self._draw_tool is None:
            self._draw_tool = LineDrawTool(canvas, self._on_line_drawn, self._end_draw)
        cur = canvas.mapTool()
        if cur is not self._draw_tool:
            self._prev_map_tool = cur
        self._draw_tool.start()
        canvas.setMapTool(self._draw_tool)
        canvas.setFocus()
        self.src_draw_radio.setChecked(True)
        self.status_label.setText("Click points on the map, right-click (or Enter) to finish, "
                                  "Backspace = undo last point, Esc = cancel.")

    def _end_draw(self):
        if self.iface is None or self._draw_tool is None:
            return
        canvas = self.iface.mapCanvas()
        if canvas.mapTool() is self._draw_tool:
            try:
                if self._prev_map_tool is not None:
                    canvas.setMapTool(self._prev_map_tool)
                else:
                    canvas.unsetMapTool(self._draw_tool)
            except Exception:
                canvas.unsetMapTool(self._draw_tool)

    def _on_line_drawn(self, canvas_pts):
        self._end_draw()
        if not self._gather_soundings():        # sets data_crs from the ticked layers
            return
        tr = self._canvas_transform(to_canvas=False)
        self._drawn_line = [QgsPointXY(tr.transform(p)) for p in canvas_pts]
        self.src_draw_radio.setChecked(True)
        self._plot_profile()

    def _pan_to_profile(self):
        if self.iface is None or len(self.profile_points) < 2:
            return
        try:
            canvas = self.iface.mapCanvas()
            tr = self._canvas_transform(True)
            pts = [tr.transform(p) for p in self.profile_points]
            ext = canvas.extent()
            if not all(ext.contains(p) for p in pts):
                cx = sum(p.x() for p in pts) / len(pts)
                cy = sum(p.y() for p in pts) / len(pts)
                canvas.setCenter(QgsPointXY(cx, cy))
                canvas.refresh()
        except Exception:
            pass

    def _lith_colors_changed(self):
        if self._has_plot and getattr(self, '_bh_drawn', None):
            self._replot_if_plotted()

    def _replot_if_plotted(self, *args):
        if self._has_plot and not getattr(self, '_loading', False):
            self._navigating = self._nav is not None   # keep the step length
            try:
                self._plot_profile()
            finally:
                self._navigating = False

    def _line_from_layer(self):
        if self.line_layer_combo.count() == 0:
            self._refresh_line_layers()
        lyr = QgsProject.instance().mapLayer(self.line_layer_combo.currentData() or '')
        if lyr is None:
            QMessageBox.warning(self, "No line layer", "Add a line layer to the project and click ↻.")
            return None, ""
        feats = list(lyr.selectedFeatures()) or list(lyr.getFeatures())
        if not feats:
            QMessageBox.warning(self, "Empty layer", f"Layer '{lyr.name()}' has no features.")
            return None, ""
        geom = QgsGeometry(feats[0].geometry())
        if geom.isMultipart():
            parts = geom.asMultiPolyline()
            pts = max(parts, key=len) if parts else []
        else:
            pts = geom.asPolyline()
        if len(pts) < 2:
            QMessageBox.warning(self, "Invalid line", "The line needs at least 2 vertices.")
            return None, ""
        data_crs = self.data_crs if (self.data_crs and self.data_crs.isValid()) else QgsProject.instance().crs()
        if lyr.crs().isValid() and data_crs.isValid() and lyr.crs() != data_crs:
            tr = QgsCoordinateTransform(lyr.crs(), data_crs, QgsProject.instance())
            pts = [tr.transform(p) for p in pts]
        txt = f"line from '{lyr.name()}'"
        if len(feats) > 1 and lyr.selectedFeatureCount() == 0:
            txt += f", first of {len(feats)} – select one to choose"
        return [QgsPointXY(p.x(), p.y()) for p in pts], txt

    def _orient(self, points):
        """Default direction W→E (or S→N for north-south lines); ⇄ Reverse flips it."""
        dx = points[-1].x() - points[0].x()
        dy = points[-1].y() - points[0].y()
        if (abs(dx) >= abs(dy) and dx < 0) or (abs(dy) > abs(dx) and dy < 0):
            points = list(reversed(points))
        if self.reverse_btn.isChecked():
            points = list(reversed(points))
        return points

    @staticmethod
    def _cardinal(az):
        names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        return names[int(((az % 360) + 22.5) // 45) % 8]

    def _direction_labels(self):
        if len(self.profile_points) < 2:
            return "A", "B"
        p0, p1 = self.profile_points[0], self.profile_points[-1]
        az = math.degrees(math.atan2(p1.x() - p0.x(), p1.y() - p0.y()))
        return self._cardinal(az + 180), self._cardinal(az)

    # -----------------------------------------------------------------
    #  Profile line, direction arrow and cursor position on the map
    # -----------------------------------------------------------------

    def _refresh_rubberband(self):
        if self.rb_line is None:
            return
        self.rb_line.reset(Qgis.GeometryType.Line)
        tr = self._canvas_transform(to_canvas=True)
        for pt in self.profile_points:
            self.rb_line.addPoint(tr.transform(pt))
        self._update_direction_marks()

    def _update_direction_marks(self, *args):
        if self.rb_arrow is None:
            return
        self.rb_arrow.reset(Qgis.GeometryType.Polygon)
        if len(self.profile_points) < 2:
            self.start_marker.hide()
            return
        tr = self._canvas_transform(to_canvas=True)
        cpts = [tr.transform(p) for p in self.profile_points]
        self.start_marker.setCenter(cpts[0])
        self.start_marker.show()
        tip, prev = cpts[-1], cpts[-2]
        dx, dy = tip.x() - prev.x(), tip.y() - prev.y()
        n = math.hypot(dx, dy)
        if n == 0:
            return
        ux, uy = dx / n, dy / n
        size = self.iface.mapCanvas().mapUnitsPerPixel() * 16
        bx, by = tip.x() - ux * size, tip.y() - uy * size
        for x, y in ((tip.x(), tip.y()),
                     (bx - uy * size * 0.5, by + ux * size * 0.5),
                     (bx + uy * size * 0.5, by - ux * size * 0.5)):
            self.rb_arrow.addPoint(QgsPointXY(x, y), False)
        self.rb_arrow.closePoints(True)

    def _on_mouse_move(self, event):
        if not self._axes or not self.show_cursor_check.isChecked():
            return
        hover = [a for a in self._axes] + ([self.ax2] if self.ax2 is not None else [])
        if event.inaxes is None or event.inaxes not in hover:
            self._hide_cursor()
            return
        ref = event.inaxes if event.inaxes in self._axes else self._axes[0]
        x, y = ref.transData.inverted().transform((event.x, event.y))
        for art in self._cursor_artists:
            try:
                art.remove()
            except Exception:
                pass
        self._cursor_artists = [a.axvline(x, color='red', linewidth=0.8) for a in self._axes]
        self._cursor_artists.append(ref.axhline(y, color='red', linewidth=0.8))
        ylab = "Elevation" if self.yaxis_combo.currentText() == "Elevation" else "Depth"
        txt = f"Distance: {x:.1f} m   {ylab}: {y:.1f} m"
        val = self._value_at(ref, x, y)
        if val:
            txt += f"   {val}"
        drawn = getattr(self, '_bh_drawn', None)
        if drawn:
            bhs_, w_, to_y_ = drawn
            elev_mode = self.yaxis_combo.currentText() == "Elevation"
            lith = lith_at(bhs_, w_, x, lambda b: (to_y_(b, np.array([0.0]))[0] - y) if elev_mode else y)
            if lith:
                txt += f"   BH {lith}"
        if self._line_geom is not None:
            d = min(max(x, 0.0), self._line_geom.length())
            pt = self._line_geom.interpolate(d).asPoint()
            txt += f"\nX: {pt.x():.1f}   Y: {pt.y():.1f}"
            if self.pos_marker is not None:
                self.pos_marker.setCenter(self._canvas_transform(True).transform(pt))
                self.pos_marker.show()
        self.cursor_label.setText(txt)
        self.canvas.draw_idle()

    def _value_at(self, ax, x, y):
        """'PHI: 123 mrad' for the layer under the cursor in panel ax (nearest sounding)."""
        dist = getattr(self, '_plot_dist', None)
        prm = next((p for p, pn in self._panels.items() if pn['ax'] is ax), None)
        if prm is None or dist is None or not len(dist):
            return ""
        lay = self._panels[prm].get('lay')
        if lay is None:
            return ""
        i = int(np.argmin(np.abs(dist - x)))
        lo = np.minimum(lay['yt'][i], lay['yb'][i])
        hi = np.maximum(lay['yt'][i], lay['yb'][i])
        j = np.where((y >= lo) & (y <= hi))[0]
        if not len(j) or not np.isfinite(lay['vals'][i, j[0]]):
            return ""
        info = PARAMS[prm]
        return f"{info['label']}: {lay['vals'][i, j[0]]:.4g} {info['unit']}".rstrip()

    def _on_mouse_leave(self, event):
        self._hide_cursor()

    def _remember_legend_pos(self, event):
        """Keep the borehole legend where the user dragged it (also after re-plotting)."""
        try:
            leg = getattr(self, '_bh_leg', None)
            if leg is not None and isinstance(leg._loc, tuple):
                self._bh_leg_pos = leg._loc
        except Exception:
            pass

    def clear_profile(self):
        """Empty the plot and remove the profile line, arrow, start and cursor markers
        from the map. Soundings/layers and settings are kept."""
        self._end_draw()
        self._has_plot = False
        self._nav = None
        self._single = False
        self._drawn_line = None
        self._line_geom = None
        self.profile_points = []
        self._cursor_artists = []
        self._panels = {}
        self._axes = []
        self.ax = None
        self.ax2 = None
        self.cursor_hline = self.cursor_vline = None
        self._last_plot_data = None
        try:
            self.figure.clf()
            self.canvas.draw_idle()
        except Exception:
            pass
        if self.rb_line is not None:
            self.rb_line.reset(Qgis.GeometryType.Line)
        if self.rb_arrow is not None:
            self.rb_arrow.reset(Qgis.GeometryType.Polygon)
        if self.start_marker is not None:
            self.start_marker.hide()
        if self.pos_marker is not None:
            self.pos_marker.hide()
        self.cursor_label.setText("X: \nY: ")
        self.status_label.setText("Profile cleared.")
        if self.iface:
            self.iface.mapCanvas().refresh()

    def _hide_cursor(self):
        for art in self._cursor_artists:
            try:
                art.remove()
            except Exception:
                pass
        self._cursor_artists = []
        if self.pos_marker is not None:
            self.pos_marker.hide()
        if self.ax:
            self.cursor_label.setText("Distance: \nX: ")
            self.canvas.draw_idle()

    # -----------------------------------------------------------------
    #  Map items life cycle
    # -----------------------------------------------------------------

    def stop_picking(self):
        """Window closed: release the drawing tool and hide the cursor marker."""
        self._end_draw()
        if self.pos_marker is not None:
            self.pos_marker.hide()

    def set_map_items_visible(self, visible):
        for item in (self.rb_line, self.rb_arrow):
            if item is not None:
                try:
                    item.setVisible(visible)
                except Exception:
                    pass
        if self.start_marker is not None:
            self.start_marker.setVisible(visible and len(self.profile_points) >= 2)
        if self.pos_marker is not None and not visible:
            self.pos_marker.hide()

    def cleanup(self):
        """Remove the profile line, arrow and markers from the map."""
        try:
            from .borehole_section import color_events
            color_events().changed.disconnect(self._lith_colors_changed)
        except Exception:
            pass
        for slot in (self._refresh_saved_profiles, self._refresh_topo_layers):
            try:
                self.registry.changed.disconnect(slot)
            except Exception:
                pass
        if getattr(self, '_edit_lyr_id', None):
            try:
                self._finish_edit(ask=True)
            except Exception:
                pass
        for lyr in self._sel_conns:
            try:
                lyr.selectionChanged.disconnect(self._update_sel_count)
            except Exception:
                pass
        self._sel_conns = []
        if not self.iface:
            return
        self._end_draw()
        canvas = self.iface.mapCanvas()
        try:
            canvas.extentsChanged.disconnect(self._update_direction_marks)
        except Exception:
            pass
        for name in ('rb_line', 'rb_arrow', 'start_marker', 'pos_marker'):
            item = getattr(self, name, None)
            if item is not None:
                try:
                    canvas.scene().removeItem(item)
                except Exception:
                    pass
                setattr(self, name, None)

    def _reset_view(self):
        home = getattr(self, '_home_lims', {})
        for a in list(self._axes) + ([self.ax2] if self.ax2 is not None else []):
            if a in home:
                a.set_xlim(*home[a][0])
                a.set_ylim(*home[a][1])
            else:
                a.autoscale()
        if self._axes:
            self.canvas.draw_idle()

    def _save_profile(self):
        fmt = self.save_format_combo.currentText()
        if self._last_plot_data is None or self._last_plot_data.empty:
            QMessageBox.warning(self, "No data", "Plot the profile first.")
            return
        if "PNG" in fmt:
            self._save_graph(".png")
        elif "map" in fmt:
            self._save_graph_with_map()
        elif "PDF" in fmt:
            self._save_graph(".pdf")
        elif "SVG" in fmt:
            self._save_graph(".svg")
        elif "CSV" in fmt:
            self._save_table(".csv", lambda p: self._last_plot_data.to_csv(p, index=False))
        elif "Excel" in fmt:
            self._save_table(".xlsx", lambda p: self._last_plot_data.to_excel(p, index=False, engine='openpyxl'))

    def _save_profile_layer(self):
        """Store the current profile line as a feature of the 'TEM-Viewer profiles' layer."""
        if not self._has_plot or len(self.profile_points) < 2:
            QMessageBox.warning(self, "No profile", "Plot a profile first.")
            return
        if self._single:
            QMessageBox.information(self, "Single sounding", "A single sounding has no profile line to save.")
            return
        proj = QgsProject.instance()
        crs = self.data_crs if (self.data_crs is not None and self.data_crs.isValid()) else proj.crs()
        from .map_export import add_layer_to_group, unique_layer_name, count_layers_of_kind
        n_existing = count_layers_of_kind(PROFILE_KIND)
        st, en = self._direction_labels()
        fid = self._profile_field_id()
        default = f"{fid} ({st}-{en})" if fid else f"Profile {n_existing + 1} ({st}-{en})"
        name, ok = QInputDialog.getText(self, "Save profile line", "Name of the profile:", text=default)
        if not ok:
            return
        name = name.strip() or default
        lname = unique_layer_name(name, PROFILE_KIND)
        lyr = QgsVectorLayer(f"LineString?crs={crs.authid()}", lname, "memory")
        lyr.dataProvider().addAttributes([QgsField(n_, t_) for n_, t_ in PROFILE_FIELDS])
        lyr.updateFields()
        lyr.setCrs(crs)
        geom, vals, warn = self._profile_values(name)
        f = QgsFeature(lyr.fields())
        f.setGeometry(geom)
        for key, val in vals.items():
            idx = lyr.fields().indexOf(key)
            if idx >= 0:
                f.setAttribute(idx, val)
        lyr.dataProvider().addFeatures([f])
        lyr.updateExtents()
        from .map_export import persist_layer
        try:
            lyr, path = persist_layer(lyr, "TEM-Viewer profiles", PROFILE_KIND)
            where = f" It is stored in {path}, so it survives reopening the project."
        except Exception as exc:
            _log(f"Saving profile to GeoPackage: {exc}")
            lyr.setCustomProperty("temviewer/kind", PROFILE_KIND)
            add_layer_to_group(lyr, "TEM-Viewer profiles")
            where = (f" WARNING: it could not be written to a file ({exc}) and is only a temporary layer - "
                     "use right-click > Make Permanent before closing the project.")
        lyr.triggerRepaint()
        # the new layer becomes the loaded saved profile, so later changes can update it
        fid = next((ft.id() for ft in lyr.getFeatures()), None)
        self._loaded_profile = (lyr.id(), fid) if fid is not None else None
        self._refresh_saved_profiles(select=lyr.id())
        self.status_label.setText(f"Profile '{name}' saved as its own layer in the group 'TEM-Viewer profiles'"
                                  "."
                                  + where + warn)

    def _profile_field_id(self):
        """Field_id of the boreholes drawn on the profile (from a 'Field_id' column of the
        borehole data; their names if there is no such column), '' when no borehole is shown."""
        drawn = getattr(self, '_bh_drawn', None)
        bhs = drawn[0] if drawn else []
        if not bhs:
            return ""
        proj = QgsProject.instance()
        ids = {}
        for lid in {b.get('layer_id') for b in bhs}:
            lyr = proj.mapLayer(lid or '')
            if lyr is None:
                continue
            try:
                ids[lid] = wlv.field_ids(self._wl_tables(lyr))
            except Exception as exc:
                _log(f"Field_id of {lyr.name()}: {exc}")
                ids[lid] = {}
        vals = []
        for b in sorted(bhs, key=lambda b_: b_['dist']):
            v = ids.get(b.get('layer_id'), {}).get(wlv.norm_name(b['name'])) or str(b['name'])
            if v not in vals:
                vals.append(v)
        txt = "-".join(vals[:3]) + ("-…" if len(vals) > 3 else "")
        return txt

    def _profile_values(self, name, keep_uid=None):
        """Geometry (data CRS, in plotted direction) and attribute values of the current plot.
        keep_uid: keep the profile's id."""
        layers = self.snd_picker.checked_layers()
        geom = QgsGeometry.fromPolylineXY(self.profile_points)
        st, en = self._direction_labels()
        try:
            tol = float(self.tolerance_edit.text())
        except ValueError:
            tol = 0.0
        n_snd = 0
        if self._last_plot_data is not None and not self._last_plot_data.empty:
            n_snd = int(self._last_plot_data['Sounding_ID'].nunique())
        uid, report_json, warn = keep_uid or "", "", ""     # no report pages in TEM-Viewer
        vals = {"name": name, "length_m": float(geom.length()), "start": st, "end": en,
                "search_w": tol, "n_snd": n_snd, "layer_ids": ";".join(l.id() for l in layers),
                "layers": "; ".join(l.name() for l in layers), "uid": uid, "report_data": report_json,
                "settings": json.dumps(self._plot_settings())}
        return geom, vals, warn

    # -----------------------------------------------------------------
    #  Option d: saved profiles - load, edit the line, update
    # -----------------------------------------------------------------
    def _plot_settings(self):
        """Everything needed to re-draw the plot as it is now (stored with a saved profile)."""
        self._cs[self._cs_param] = self._widget_cs()
        return dict(
            yaxis=self.yaxis_combo.currentText(), barwidth=self.barwidth_edit.text(),
            bar_unit=self.barunit_combo.currentData(),
            wl=self.wl_check.isChecked(), wl_col=self.wl_combo.currentData(),
            wl_mode=self.wl_mode_combo.currentData(),
            topo=self.topo_check.isChecked(), topo_layer=self.topo_combo.currentData(),
            interp=self.interp_check.isChecked(), frames=self.frames_check.isChecked(),
            params=self._selected_params(),
            doi_mask=self.doi_mask_check.isChecked(), doi_col=self.doi_col_combo.currentText(),
            doi_opacity=self.doi_opacity_spin.value(),
            misfit=self.misfit_check.isChecked(), misfit_col=self.misfit_col_combo.currentText(),
            misfit_max=self.misfit_max_spin.value(), front=self.front_edit.text(),
            bh=self.bh_check.isChecked(), bh_dist=self.bh_dist_spin.value(),
            bh_width=self.bh_width_spin.value(), bh_ids=self.bh_picker.checked_ids(),
            cs={k: dict(v) for k, v in self._cs.items()}, cs_param=self._cs_param)

    def _apply_settings(self, s):
        """Restore the plot settings of a saved profile (no re-plots while doing so)."""
        def _try(fn):
            try:
                fn()
            except Exception as exc:
                _log(f"Restoring setting: {exc}")
        if 'yaxis' in s:
            _try(lambda: self.yaxis_combo.setCurrentText(s['yaxis']))
        if s.get('bar_unit') in ('m', 'px'):
            def _unit():
                self.barunit_combo.blockSignals(True)
                self.barunit_combo.setCurrentIndex(self.barunit_combo.findData(s['bar_unit']))
                self.barunit_combo.blockSignals(False)
                self._bw_unit = s['bar_unit']
            _try(_unit)
        if 'barwidth' in s:
            _try(lambda: self.barwidth_edit.setText(str(s['barwidth'])))
        if s.get('wl_mode') in ('depth', 'elev'):
            _try(lambda: self.wl_mode_combo.setCurrentIndex(self.wl_mode_combo.findData(s['wl_mode'])))
        if s.get('topo_layer'):
            def _topo():
                k = self.topo_combo.findData(s['topo_layer'])
                if k >= 0:
                    self.topo_combo.setCurrentIndex(k)
            _try(_topo)
        if 'topo' in s:
            _try(lambda: self.topo_check.setChecked(bool(s['topo']) and self.topo_combo.currentData() is not None))
        if 'wl' in s:
            def _wl():
                self.wl_check.blockSignals(True)
                self.wl_check.setChecked(bool(s['wl']))
                self.wl_check.blockSignals(False)
                self._refresh_wl_columns()
                if s.get('wl_col'):
                    k = self._wl_index(s['wl_col'])
                    if k >= 0:
                        self.wl_combo.setCurrentIndex(k)
            _try(_wl)
        for key, w in (('interp', self.interp_check), ('frames', self.frames_check),
                       ('doi_mask', self.doi_mask_check), ('misfit', self.misfit_check)):
            if key in s:
                _try(lambda w=w, key=key: w.setChecked(bool(s[key])))
        for key, w in (('doi_col', self.doi_col_combo), ('misfit_col', self.misfit_col_combo)):
            if key in s:
                _try(lambda w=w, key=key: w.setCurrentText(s[key]))
        for key, w in (('doi_opacity', self.doi_opacity_spin), ('misfit_max', self.misfit_max_spin),
                       ('bh_dist', self.bh_dist_spin), ('bh_width', self.bh_width_spin)):
            if key in s:
                _try(lambda w=w, key=key: w.setValue(float(s[key])))
        if 'front' in s:
            _try(lambda: self.front_edit.setText(str(s['front'] or "")))
        if 'params' in s:
            for prm, cb in self.param_checks.items():
                _try(lambda cb=cb, prm=prm: cb.setChecked(prm in s['params']))
        if 'bh' in s:
            _try(lambda: self.bh_check.setChecked(bool(s['bh'])))
        if s.get('bh_ids'):
            _try(lambda: self.bh_picker.set_checked_ids(s['bh_ids']))
        if isinstance(s.get('cs'), dict):
            def _cs():
                for k, v in s['cs'].items():
                    if k in PARAMS and isinstance(v, dict) and \
                            {'cmap', 'rev', 'log', 'vmin', 'vmax'} <= set(v):
                        self._cs[k] = dict(v)
                prm = s.get('cs_param') or self._cs_param
                k = self.cs_param_combo.findData(prm)
                if k >= 0:
                    self.cs_param_combo.blockSignals(True)
                    self.cs_param_combo.setCurrentIndex(k)
                    self.cs_param_combo.blockSignals(False)
                else:
                    prm = self.cs_param_combo.currentData() or 'RHO'
                self._cs_param = prm
                self._load_cs(prm)
            _try(_cs)

    def _refresh_saved_profiles(self, *args, select=None):
        proj = QgsProject.instance()
        cur = select or self.saved_combo.currentData()
        layers = [l for l in proj.mapLayers().values()
                  if isinstance(l, QgsVectorLayer) and l.customProperty("temviewer/kind") == PROFILE_KIND]
        layers.sort(key=lambda l: l.name().lower())
        self.saved_combo.blockSignals(True)
        self.saved_combo.clear()
        for l in layers:
            empty = l.isValid() and l.featureCount() == 0
            self.saved_combo.addItem(l.name() + ("  [empty]" if empty else ""), l.id())
        if not layers:
            self.saved_combo.addItem("(no saved profiles)", None)
        k = self.saved_combo.findData(cur) if cur else -1
        self.saved_combo.setCurrentIndex(k if k >= 0 else 0)
        self.saved_combo.blockSignals(False)
        if self._loaded_profile and proj.mapLayer(self._loaded_profile[0]) is None:
            self._loaded_profile = None
        if getattr(self, '_edit_lyr_id', None) and proj.mapLayer(self._edit_lyr_id) is None:
            self._edit_lyr_id = None
            self.saved_edit_btn.blockSignals(True)
            self.saved_edit_btn.setChecked(False)
            self.saved_edit_btn.blockSignals(False)
        self._update_saved_ui()

    def _on_saved_combo_changed(self, *args):
        if self.saved_combo.currentData():
            if self._edit_lyr_id and self._edit_lyr_id != self.saved_combo.currentData():
                self._finish_edit(ask=True)
            self.src_saved_radio.setChecked(True)
        self._update_saved_ui()

    def _saved_layer(self):
        return QgsProject.instance().mapLayer(self.saved_combo.currentData() or '')

    def _loaded_layer(self):
        if not self._loaded_profile:
            return None
        return QgsProject.instance().mapLayer(self._loaded_profile[0])

    def _update_saved_ui(self):
        lyr = self._saved_layer()
        loaded = self._loaded_layer()
        self.saved_load_btn.setEnabled(lyr is not None)
        self.saved_edit_btn.setEnabled(lyr is not None and self.iface is not None)
        self.saved_update_btn.setEnabled(loaded is not None)
        if loaded is not None:
            tmp = "  (temporary layer – not saved in a file)" if loaded.providerType() == 'memory' else ""
            self.saved_info_label.setText(f"Loaded: {loaded.name()}{tmp}")
        elif lyr is not None and lyr.featureCount() == 0:
            self.saved_info_label.setText("This saved profile is empty: it was a temporary layer that lost its "
                                          "content when the project was closed. Plot the profile again and "
                                          "save it; it is now stored in a file.")
        else:
            self.saved_info_label.setText("")

    def _saved_feature(self, lyr):
        if self._loaded_profile and self._loaded_profile[0] == lyr.id():
            f = lyr.getFeature(self._loaded_profile[1])
            if f.isValid():
                return f
        feats = list(lyr.selectedFeatures()) or list(lyr.getFeatures())
        return feats[0] if feats else None

    def _line_from_saved(self):
        lyr = self._saved_layer()
        if lyr is None:
            QMessageBox.warning(self, "No saved profile",
                                "Save a profile with 'Save Line as Layer' first, or pick one in the list.")
            return None, ""
        f = self._saved_feature(lyr)
        if f is None or f.geometry() is None or f.geometry().isEmpty():
            QMessageBox.warning(self, "Empty profile",
                                f"'{lyr.name()}' has no line. It was probably a temporary layer that lost its "
                                "content when the project was closed – plot the profile again and save it.")
            return None, ""
        geom = QgsGeometry(f.geometry())
        if geom.isMultipart():
            parts = geom.asMultiPolyline()
            pts = max(parts, key=len) if parts else []
        else:
            pts = geom.asPolyline()
        if len(pts) < 2:
            QMessageBox.warning(self, "Invalid line", "The saved line needs at least 2 vertices.")
            return None, ""
        data_crs = self.data_crs if (self.data_crs and self.data_crs.isValid()) else QgsProject.instance().crs()
        if lyr.crs().isValid() and data_crs.isValid() and lyr.crs() != data_crs:
            tr = QgsCoordinateTransform(lyr.crs(), data_crs, QgsProject.instance())
            pts = [tr.transform(p) for p in pts]
        return [QgsPointXY(p.x(), p.y()) for p in pts], f"saved profile '{lyr.name()}'"

    def _load_saved_profile(self):
        """Restore a saved profile: its sounding layers, search width and plot settings, then plot it."""
        lyr = self._saved_layer()
        if lyr is None:
            QMessageBox.information(self, "Saved profiles", "There is no saved profile to load.")
            return False
        if self._edit_lyr_id and self._edit_lyr_id != lyr.id():
            self._finish_edit(ask=True)
        if self._loaded_profile and self._loaded_profile[0] != lyr.id():
            self._loaded_profile = None
        f = self._saved_feature(lyr)
        if f is None:
            self._update_saved_ui()
            QMessageBox.warning(self, "Empty profile",
                                f"'{lyr.name()}' has no line. It was probably a temporary layer that lost its "
                                "content when the project was closed – plot the profile again and save it.")
            return False
        names = lyr.fields().names()
        proj = QgsProject.instance()
        snd_layers = {l.id(): l for l in self.registry.sounding_layers()}
        ids, missing = [], []
        if 'layer_ids' in names and f['layer_ids']:
            ids = [i for i in str(f['layer_ids']).split(';') if i in snd_layers]
        if not ids and 'layers' in names and f['layers']:
            by_name = {l.name(): i for i, l in snd_layers.items()}
            wanted = [n.strip() for n in str(f['layers']).split(';') if n.strip()]
            ids = [by_name[n] for n in wanted if n in by_name]
            missing = [n for n in wanted if n not in by_name]
        settings = {}
        if 'settings' in names and f['settings']:
            try:
                settings = json.loads(str(f['settings']))
            except Exception:
                settings = {}
        self._loading = True
        try:
            if ids:
                self.snd_picker.set_checked_ids(ids)
            if 'search_w' in names and f['search_w'] not in (None, ''):
                try:
                    self.tolerance_edit.setText(f"{float(f['search_w']):g}")
                except (TypeError, ValueError):
                    pass
            self._apply_settings(settings)
            self.reverse_btn.blockSignals(True)
            self.reverse_btn.setChecked(False)       # the stored line already has its direction
            self.reverse_btn.blockSignals(False)
            self.src_saved_radio.setChecked(True)
        finally:
            self._loading = False
        self._loaded_profile = (lyr.id(), f.id())
        self._update_saved_ui()
        if not self.snd_picker.checked_layers():
            QMessageBox.warning(self, "Sounding layers missing",
                                "The sounding layers of this profile are not in the project"
                                + (f" ({'; '.join(missing)})" if missing else "")
                                + ". Add them and tick them under 1. Sounding layers, then Plot Profile.")
            return False
        self._plot_profile()
        if not self._has_plot:
            return False
        self._pan_to_profile()
        self.status_label.setText(
            f"Loaded saved profile '{lyr.name()}'"
            + (f" (not found: {'; '.join(missing)})" if missing else "")
            + ". Change the settings or the line (Edit line, or draw a new one with a / b / c), then "
              "'Update saved profile' to store the changes – or 'Save Line as Layer' to keep them as a new profile.")
        return True

    # ---- editing the saved line on the map --------------------------
    def _toggle_edit_saved(self, on):
        if not on:
            self._finish_edit(ask=True)
            return
        lyr = self._saved_layer()
        ok = lyr is not None and self.iface is not None
        if ok and self._loaded_layer() is not lyr:
            ok = self._load_saved_profile()
        if ok and not lyr.isEditable():
            ok = lyr.startEditing()
        if not ok:
            self.saved_edit_btn.blockSignals(True)
            self.saved_edit_btn.setChecked(False)
            self.saved_edit_btn.blockSignals(False)
            if lyr is not None and self.iface is not None and not lyr.isEditable() and self._loaded_layer() is lyr:
                QMessageBox.warning(self, "Edit line", f"'{lyr.name()}' cannot be edited (read-only file?).")
            return
        self.src_saved_radio.setChecked(True)
        self._edit_lyr_id = lyr.id()
        lyr.geometryChanged.connect(self._on_saved_geom_signal)
        lyr.editingStopped.connect(self._on_edit_stopped_outside)
        try:
            self.iface.setActiveLayer(lyr)
            self.iface.actionVertexTool().trigger()
        except Exception as exc:
            _log(f"Vertex tool: {exc}")
        self.status_label.setText(
            "Editing the saved line: drag vertices, double-click a segment to add one, select a vertex and "
            "press Delete to remove it. The profile is re-plotted after each change. "
            "Click 'Edit line' again to finish.")

    def _disconnect_edit(self, lyr):
        for sig, slot in ((lyr.geometryChanged, self._on_saved_geom_signal),
                          (lyr.editingStopped, self._on_edit_stopped_outside)):
            try:
                sig.disconnect(slot)
            except Exception:
                pass

    def _on_saved_geom_signal(self, *args):
        self._geom_timer.start()

    def _on_saved_geom_edited(self):
        if self._edit_lyr_id and self.src_saved_radio.isChecked() and \
                self.saved_combo.currentData() == self._edit_lyr_id:
            self._plot_profile()

    def _on_edit_stopped_outside(self):
        """Editing was ended with the QGIS toolbar (QGIS already asked to save or discard)."""
        lyr = QgsProject.instance().mapLayer(self._edit_lyr_id or '')
        if lyr is not None:
            self._disconnect_edit(lyr)
        self._edit_lyr_id = None
        self.saved_edit_btn.blockSignals(True)
        self.saved_edit_btn.setChecked(False)
        self.saved_edit_btn.blockSignals(False)
        if self.src_saved_radio.isChecked() and self._has_plot:
            self._plot_profile()

    def _finish_edit(self, ask=True):
        lyr = QgsProject.instance().mapLayer(self._edit_lyr_id or '')
        self._edit_lyr_id = None
        self.saved_edit_btn.blockSignals(True)
        self.saved_edit_btn.setChecked(False)
        self.saved_edit_btn.blockSignals(False)
        if lyr is None:
            return
        self._disconnect_edit(lyr)
        if not lyr.isEditable():
            return
        if lyr.isModified() and ask:
            ans = QMessageBox.question(
                self, "Edit line", f"Keep the changes to the line of '{lyr.name()}'?\n\n"
                "Yes = store the new line and update the saved profile with the current plot\n"
                "No = discard the changes",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes)
            if ans == QMessageBox.StandardButton.Yes:
                self._update_saved_profile(confirm=False)
                return
        lyr.rollBack()
        lyr.triggerRepaint()
        if self.src_saved_radio.isChecked() and self._has_plot:
            self._plot_profile()

    def _update_saved_profile(self, confirm=True):
        """Overwrite the loaded saved profile with the current plot."""
        lyr = self._loaded_layer()
        if lyr is None:
            QMessageBox.information(self, "Update saved profile",
                                    "Load a saved profile first (option d), or save this one with "
                                    "'Save Line as Layer'.")
            return
        if not self._has_plot or len(self.profile_points) < 2 or self._single:
            QMessageBox.warning(self, "No profile", "Plot a profile (a line, not a single sounding) first.")
            return
        fid = self._loaded_profile[1]
        f = lyr.getFeature(fid)
        if not f.isValid():
            QMessageBox.warning(self, "Update saved profile", f"The profile in '{lyr.name()}' was not found.")
            return
        names = lyr.fields().names()
        name = str(f['name']) if 'name' in names and f['name'] not in (None, '') else lyr.name()
        if confirm and QMessageBox.question(
                self, "Update saved profile",
                f"Replace the saved profile '{name}' with the current plot "
                "(line, sounding layers and plot settings)?") \
                != QMessageBox.StandardButton.Yes:
            return
        if lyr.isEditable():
            self._disconnect_edit(lyr)
            if not lyr.commitChanges():
                err = "; ".join(lyr.commitErrors())
                lyr.rollBack()
                QMessageBox.critical(self, "Update saved profile", f"Could not store the edited line: {err}")
                return
            if self._edit_lyr_id == lyr.id():
                self._edit_lyr_id = None
            self.saved_edit_btn.blockSignals(True)
            self.saved_edit_btn.setChecked(False)
            self.saved_edit_btn.blockSignals(False)
        old_uid = str(f['uid']) if 'uid' in names and f['uid'] not in (None, '') else None
        geom, vals, warn = self._profile_values(name, keep_uid=old_uid)
        data_crs = self.data_crs if (self.data_crs and self.data_crs.isValid()) else QgsProject.instance().crs()
        if lyr.crs().isValid() and data_crs.isValid() and lyr.crs() != data_crs:
            geom.transform(QgsCoordinateTransform(data_crs, lyr.crs(), QgsProject.instance()))
        pr = lyr.dataProvider()
        missing = [QgsField(n_, t_) for n_, t_ in PROFILE_FIELDS if lyr.fields().indexOf(n_) < 0]
        if missing:
            pr.addAttributes(missing)
            lyr.updateFields()
        attrs = {lyr.fields().indexOf(k): v for k, v in vals.items() if lyr.fields().indexOf(k) >= 0}
        ok = pr.changeGeometryValues({fid: geom}) and pr.changeAttributeValues({fid: attrs})
        if not ok:
            QMessageBox.critical(self, "Update saved profile",
                                 f"'{lyr.name()}' could not be written (is the file read-only or open elsewhere?).")
            return
        lyr.updateExtents()
        lyr.triggerRepaint()
        # the stored line now runs in the plotted direction
        self.reverse_btn.blockSignals(True)
        self.reverse_btn.setChecked(False)
        self.reverse_btn.blockSignals(False)
        self._refresh_saved_profiles(select=lyr.id())
        self.src_saved_radio.setChecked(True)
        tmp = (" Note: this is a temporary layer – right-click > Make Permanent before closing the project."
               if lyr.providerType() == 'memory' else "")
        self.status_label.setText(f"Saved profile '{name}' updated." + tmp + warn)

    def _build_report_item(self, name, orientations=None):
        """Render what the A4 pages need, now: the location map and the profile image
        for each page format (landscape / portrait), plus the facts of this profile."""
        from .map_export import build_report_item
        line = np.array([[p_.x(), p_.y()] for p_ in self.profile_points], dtype=float)
        drawn = getattr(self, '_bh_drawn', None)
        st, en = self._direction_labels()
        try:
            tol = float(self.tolerance_edit.text())
        except ValueError:
            tol = 0.0
        n_snd = 0
        if self._last_plot_data is not None and not self._last_plot_data.empty:
            n_snd = int(self._last_plot_data['Sounding_ID'].nunique())
        crs_id = self.data_crs.authid() if (self.data_crs is not None and self.data_crs.isValid()) else ""
        bh_on = getattr(self, 'bh_check', None) is not None and self.bh_check.isChecked()
        item = build_report_item(
            self.figure, name, line, self.data_crs, self.iface, (st, en),
            soundings_xy=getattr(self, '_plot_xy', None), boreholes=drawn[0] if drawn else None,
            bh_dist=self.bh_dist_spin.value() if bh_on else None, orientations=orientations,
            before=self._hide_cursor, kind='profile',
            length_m=float(QgsGeometry.fromPolylineXY(self.profile_points).length()),
            n_snd=n_snd, search_w=tol, crs=crs_id,
            layers="; ".join(l.name() for l in self.snd_picker.checked_layers()),
            yaxis=self.yaxis_combo.currentText(),
            params=" + ".join(PARAMS[p_]['label'] for p_ in self._selected_params()))
        self.canvas.draw_idle()
        return item

    def _save_graph_with_map(self):
        """A4 PDF: map of the profile line (canvas layers incl. background map) above the plot."""
        if self._single or len(self.profile_points) < 2:
            QMessageBox.information(self, "Location map", "Plot a profile (not a single sounding) first.")
            return
        choices = ["A4 landscape", "A4 portrait"]
        cur = 0 if getattr(self, '_page_orient', 'landscape') == 'landscape' else 1
        choice, ok = QInputDialog.getItem(self, "Graph + location map", "Page format:", choices, cur, False)
        if not ok:
            return
        orient = 'landscape' if choice == choices[0] else 'portrait'
        self._page_orient = orient
        path, _ = QFileDialog.getSaveFileName(self, "Save graph with location map", "", "PDF (*.pdf)")
        if not path:
            return
        if not path.lower().endswith('.pdf'):
            path += '.pdf'
        from .map_export import save_item_pdf
        from qgis.PyQt.QtWidgets import QApplication
        st, en = self._direction_labels()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            item = self._build_report_item(f"Profile {st} → {en}", [orient])
            save_item_pdf(path, item, orient)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.status_label.setText(f"Graph with location map ({choice}) saved to {path}")

    def _save_graph(self, ext):
        path, _ = QFileDialog.getSaveFileName(self, "Save graph", "", f"{ext.upper()} (*{ext})")
        if not path:
            return
        if not path.lower().endswith(ext):
            path += ext
        self.figure.savefig(path, dpi=300, bbox_inches='tight')
        self.status_label.setText(f"Graph saved to {path}")

    def _save_table(self, ext, writer):
        path, _ = QFileDialog.getSaveFileName(self, "Save data", "", f"{ext.upper()} (*{ext})")
        if not path:
            return
        if not path.lower().endswith(ext):
            path += ext
        try:
            writer(path)
            self.status_label.setText(f"Data saved to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _draw_profile(self, only_ids=None):
        if self.models_df is None or self.models_df.empty:
            QMessageBox.warning(self, "No data", "No soundings in the ticked layers.")
            return
        if len(self.profile_points) < 2:
            QMessageBox.warning(self, "Need line", "The profile line needs at least 2 points.")
            return

        line_geom = QgsGeometry.fromPolylineXY(self.profile_points)
        try:
            tolerance = float(self.tolerance_edit.text())
            bar_value = float(self.barwidth_edit.text().replace(',', '.'))
            if bar_value <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Invalid input", "Check search width and bar width.")
            return
        bar_width = bar_value
        bar_px = self.barunit_combo.currentData() == 'px'

        interp = self.interp_check.isChecked()
        y_mode = self.yaxis_combo.currentText().lower()
        coord = 'depth' if y_mode == 'depth' else 'elevation'
        x_col, y_col, elev_col, id_col = 'X', 'Y', 'ELEVATION', 'RECORD'

        # Project soundings
        df_pts = self.models_df[[x_col, y_col, id_col]].copy()
        df_pts['_row'] = np.arange(len(df_pts))       # position in models_df (RECORD may repeat)
        df_pts[x_col] = pd.to_numeric(df_pts[x_col], errors='coerce')
        df_pts[y_col] = pd.to_numeric(df_pts[y_col], errors='coerce')
        df_pts = df_pts.dropna()
        if only_ids is not None:
            df_pts = df_pts[df_pts[id_col].astype(float).isin(only_ids)]

        recs = []
        for _, row in df_pts.iterrows():
            pt_geom = QgsGeometry.fromPointXY(QgsPointXY(float(row[x_col]), float(row[y_col])))
            offset = pt_geom.distance(line_geom)
            if only_ids is None and offset > tolerance:
                continue
            recs.append({
                '_row': int(row['_row']),
                id_col: row[id_col],
                'distance': float(line_geom.lineLocatePoint(pt_geom)),
                'offset': float(offset),
            })

        if not recs:
            QMessageBox.information(self, "No soundings", f"No soundings within {tolerance} m.")
            return

        prof_df = pd.DataFrame(recs).sort_values('distance').reset_index(drop=True)
        n_snd = len(prof_df)
        if bar_px:
            # width as a share of the section: bar = f * (span + 2 bar), f = px / 1000 px
            f = min(bar_value / BAR_REF_PX, 0.45)
            span = float(prof_df['distance'].max() - prof_df['distance'].min())
            if span <= 0:                 # one sounding (or all at one place): use the line length
                span = float(line_geom.length())
            bar_width = max(f * span / (1.0 - 2.0 * f), 1e-6) if span > 0 else 10.0

        if interp and n_snd < 2:
            interp = False          # a single sounding is always drawn as one column

        # ---- settings ------------------------------------------------------
        df = self.models_df
        params = self._selected_params()
        doi_idx = self.doi_col_combo.currentIndex()
        mask_doi = self.doi_mask_check.isChecked()
        mask_alpha = self.doi_opacity_spin.value() / 100.0
        hide_below_doi = mask_doi and mask_alpha >= 0.999
        misfit_col = self.misfit_col_combo.currentText() if self.misfit_check.isChecked() else None
        if misfit_col and misfit_col not in df.columns:
            misfit_col = None

        # ---- model arrays of the profile soundings (vectorised) -------------
        rows = df.iloc[prof_df['_row'].to_numpy()]
        try:
            top_d, bot_d = layer_geometry(rows)
        except ValueError as exc:
            QMessageBox.warning(self, "Model layers", str(exc))
            return
        elev = pd.to_numeric(rows[elev_col], errors='coerce').to_numpy(dtype=float)
        dist = prof_df['distance'].to_numpy(dtype=float)
        ns, L = top_d.shape
        to_y = (lambda d: elev[:, None] - d) if coord == 'elevation' else (lambda d: d)
        misfit = (pd.to_numeric(rows[misfit_col], errors='coerce').to_numpy(dtype=float)
                  if misfit_col else np.full(ns, np.nan))

        def doi_of(prm):
            col = (find_doi_column(rows.columns, PARAMS[prm]['doi'], doi_idx)
                   or find_doi_column(rows.columns, PARAMS['RHO']['doi'], doi_idx))
            if col is None:
                return np.full(ns, np.nan)
            d = pd.to_numeric(rows[col], errors='coerce').to_numpy(dtype=float)
            return np.where(d > 0, d, np.nan)

        layers = {}
        layers_full = {prm: layer_values(rows, prm)[:, :L] for prm in params}
        y_lo, y_hi = [], []
        for prm in params:
            vals = layer_values(rows, prm)[:, :L]
            doi_d = doi_of(prm)
            t, b_ = top_d.copy(), bot_d.copy()
            if hide_below_doi:
                lim = np.where(np.isfinite(doi_d), doi_d, np.inf)[:, None]
                vals = np.where(t < lim, vals, np.nan)
                b_ = np.minimum(b_, lim)
            if self._cs_for(prm) is None or (prm != 'RHO' and prm not in self._cs):
                self._cs[prm] = self._default_cs(prm, vals)
                if prm == self._cs_param:
                    self._load_cs(prm)
            yt, yb = to_y(t), to_y(b_)
            ok = np.isfinite(vals)
            layers[prm] = dict(vals=vals, yt=yt, yb=yb, doi=doi_d, ok=ok)
            if ok.any():
                y_lo.append(np.nanmin(np.where(ok, np.minimum(yt, yb), np.nan)))
                y_hi.append(np.nanmax(np.where(ok, np.maximum(yt, yb), np.nan)))
        if not y_lo:
            QMessageBox.information(self, "No layers", "No valid model layers for these soundings.")
            return
        y_min, y_max = float(min(y_lo)), float(max(y_hi))
        pad = max(abs(y_max - y_min) * 0.05, 2.0)
        # visible y-range: 20 m above the highest surface, 20 m below the deepest DOI
        # (of the plotted parameters); without a DOI the whole model is shown
        MARGIN = 20.0
        all_doi = np.concatenate([layers[p_]['doi'] for p_ in params])
        if coord == 'elevation':
            top_y = np.nanmax(elev) if np.isfinite(elev).any() else y_max
            view_hi = float(top_y) + MARGIN
            doi_el = np.concatenate([elev - layers[p_]['doi'] for p_ in params])
            view_lo = (float(np.nanmin(doi_el)) - MARGIN if np.isfinite(doi_el).any()
                       else y_min - pad)
        else:
            view_lo = -MARGIN
            view_hi = (float(np.nanmax(all_doi)) + MARGIN if np.isfinite(all_doi).any()
                       else y_max + pad)
        surf_y = elev if coord == 'elevation' else np.zeros(ns)

        # topography along the line from a raster (DEM)
        topo, topo_msg = None, ""
        if self.topo_check.isChecked() and not self._single:
            if coord != 'elevation':
                topo_msg = "topography is drawn with the Elevation y-axis only"
            else:
                try:
                    topo, topo_msg = self._sample_topography(line_geom)
                except Exception as exc:
                    topo_msg = f"topography could not be sampled: {exc}"
                    _log(topo_msg)
                if topo is not None:
                    ok_t = np.isfinite(topo[1])
                    if ok_t.any():
                        view_hi = max(view_hi, float(np.nanmax(topo[1])) + 10.0)
                    else:
                        topo, topo_msg = None, "the raster has no values along this line"
        topo_on = topo is not None

        # ---- figure: one panel per parameter ---------------------------------
        self.figure.clf()
        n_p = len(params)
        self._axes = []
        self._panels = {}
        self._cursor_artists = []
        self.ax2 = None
        self.cursor_hline = self.cursor_vline = None
        show_misfit = bool(misfit_col) and np.isfinite(misfit).any()
        x_lo = dist.min() - bar_width
        x_hi = dist.max() + bar_width
        rows_for_table = []
        # boreholes within the chosen distance of the line
        bhs, bh_w = [], 0.0
        self._bh_drawn = None
        if getattr(self, 'bh_check', None) is not None and self.bh_check.isChecked() and not self._single:
            try:
                line_xy = np.array([[p_.x(), p_.y()] for p_ in self.profile_points], dtype=float)
                bhs = boreholes_near_line(self.registry, self.bh_picker.checked_layers(), line_xy,
                                          self.data_crs, self.bh_dist_spin.value())
            except Exception as exc:
                _log(f"Boreholes on profile: {exc}")
                bhs = []
            try:
                bh_w = float(self.bh_width_spin.value())
            except (AttributeError, TypeError, ValueError):
                bh_w = 0.0
            if not bh_w > 0:
                bh_w = max(bar_width * 0.8, (x_hi - x_lo) * 0.012)
            if bhs:     # boreholes beyond the line ends are drawn at their projected distance
                x_lo = min(x_lo, min(b['dist'] for b in bhs) - bh_w)
                x_hi = max(x_hi, max(b['dist'] for b in bhs) + bh_w)
        ground_at = (lambda d_: float(np.interp(d_, dist, elev))) if ns > 1 else (lambda d_: float(elev[0]))

        def bh_y(b, depths):
            if coord != 'elevation':
                return depths
            e = b['elev'] if np.isfinite(b['elev']) else ground_at(b['dist'])
            return e - depths

        # water level of each borehole (from the chosen column)
        wl_on, n_wl, wl_msg = False, 0, ""
        if bhs and self.wl_check.isChecked() and self.wl_combo.currentData():
            try:
                wl_mode = self.wl_mode_combo.currentData()
                wl_vals, wl_msg = self._water_levels(bhs)
                for b in bhs:
                    b['wl'] = wl_vals.get((b.get('layer_id'), b['name']), np.nan)
                    b['wl_mode'] = wl_mode
                wl_on = any(np.isfinite(b['wl']) for b in bhs)
            except Exception as exc:
                wl_msg = f"Water level: {exc}"
                _log(wl_msg)

        def wl_y(b):
            """y of the water level of borehole b in the current plot (nan if unknown)."""
            v = b.get('wl', np.nan)
            if not np.isfinite(v):
                return np.nan
            e = b['elev'] if np.isfinite(b['elev']) else ground_at(b['dist'])
            if b.get('wl_mode') == 'elev':
                return v if coord == 'elevation' else e - v
            return e - v if coord == 'elevation' else v
        lith_used = {}
        from .map_export import panel_gridspec, horizontal_colorbar
        gs = panel_gridspec(self.figure, n_p)
        for k, prm in enumerate(params):
            ax = self.figure.add_subplot(gs[2 * k, :], sharex=self._axes[0] if self._axes else None)
            self._axes.append(ax)
            cmap, norm, err = self._cmap_norm_for(prm, layers[prm]['vals'])
            if err:
                QMessageBox.warning(self, "Color scale", err)
                return
            lay = layers[prm]
            vals = lay['vals'].copy()
            if isinstance(norm, matplotlib.colors.LogNorm):
                vals = np.where(vals > 0, vals, np.nan)
            artist = None
            if not interp:
                left = np.repeat((dist - bar_width / 2.0)[:, None], L, axis=1)
                y0, y1 = np.minimum(lay['yt'], lay['yb']), np.maximum(lay['yt'], lay['yb'])
                keep = np.isfinite(y0) & np.isfinite(y1) & (y1 > y0)
                if hide_below_doi:
                    keep &= np.isfinite(vals)
                if keep.any():
                    keep_idx = np.nonzero(keep)
                    try:                       # a chosen sounding is drawn last = in front
                        front = float(self.front_edit.text().strip())
                        is_front = (pd.to_numeric(rows['_SRC_RECORD'], errors='coerce').to_numpy(float)
                                    == front)[keep_idx[0]]
                        o = np.argsort(is_front, kind='stable')
                        keep_idx = (keep_idx[0][o], keep_idx[1][o])
                    except Exception:
                        pass
                    lx, b0, b1 = left[keep_idx], y0[keep_idx], y1[keep_idx]
                    verts = np.stack([np.stack([lx, b0], 1), np.stack([lx + bar_width, b0], 1),
                                      np.stack([lx + bar_width, b1], 1), np.stack([lx, b1], 1)], axis=1)
                    artist = matplotlib.collections.PolyCollection(verts, cmap=cmap, norm=norm,
                                                                   edgecolors='none', linewidths=0)
                    artist.set_array(np.ma.masked_invalid(vals[keep_idx].astype(float)))
                    ax.add_collection(artist)
                if getattr(self, 'frames_check', None) is not None and self.frames_check.isChecked():
                    ft, fb = ((np.where(np.isfinite(vals), lay['yt'], np.nan),
                               np.where(np.isfinite(vals), lay['yb'], np.nan))
                              if hide_below_doi else (lay['yt'], lay['yb']))
                    draw_sounding_frames(ax, dist - bar_width / 2.0, dist + bar_width / 2.0, ft, fb)
            else:
                ny = max(int((y_max - y_min) / 1.0), 100)
                yi = np.linspace(y_min, y_max, ny + 1)
                yc = (yi[:-1] + yi[1:]) / 2.0
                lo_ = np.minimum(lay['yt'], lay['yb'])
                hi_ = np.maximum(lay['yt'], lay['yb'])
                logi = isinstance(norm, matplotlib.colors.LogNorm)
                vv = np.log10(vals) if logi else vals
                R = np.full((ny, ns), np.nan)
                for j in range(L):
                    m = (yc[:, None] > lo_[None, :, j]) & (yc[:, None] <= hi_[None, :, j])
                    R = np.where(m, vv[None, :, j], R)
                nx = int(min(max((x_hi - x_lo) / max(1.0, bar_width / 2.0), 100), 4000))
                xi = np.linspace(x_lo, x_hi, nx + 1)
                xc = (xi[:-1] + xi[1:]) / 2.0
                if ns >= 2:
                    idx = np.clip(np.searchsorted(dist, xc) - 1, 0, ns - 2)
                    span = dist[idx + 1] - dist[idx]
                    a_ = np.where(span > 0, (xc - dist[idx]) / np.where(span > 0, span, 1), 0.0)
                    inside = (xc >= dist[0]) & (xc <= dist[-1])
                    Z = R[:, idx] * (1 - a_) + R[:, idx + 1] * a_
                    Z[:, ~inside] = np.nan
                else:
                    Z = np.repeat(R, nx, axis=1)
                if logi:
                    Z = 10 ** Z
                artist = ax.pcolormesh(xi, yi, np.ma.masked_invalid(Z), shading='flat', cmap=cmap, norm=norm)
                if getattr(self, 'frames_check', None) is not None and self.frames_check.isChecked():
                    ft, fb = ((np.where(np.isfinite(vals), lay['yt'], np.nan),
                               np.where(np.isfinite(vals), lay['yb'], np.nan))
                              if hide_below_doi else (lay['yt'], lay['yb']))
                    draw_sounding_frames(ax, dist - bar_width / 2.0, dist + bar_width / 2.0, ft, fb)
            ax.plot(dist, surf_y, 'kv', markersize=4 if ns <= 150 else 2, zorder=5)
            if topo_on:
                ax.plot(topo[0], topo[1], color=TOPO_COLOR, linewidth=1.4, zorder=7, solid_joinstyle='round')
            ax.set_xlim(x_lo, x_hi)

            # DOI line + mask
            doi_y = to_y(lay['doi'][:, None])[:, 0] if coord == 'elevation' else lay['doi']
            okd = np.isfinite(doi_y)
            y_floor = min(y_min - pad, view_lo) if coord == 'elevation' else max(y_max + pad, view_hi)
            if mask_doi and not okd.any():
                ax.text(0.5, 0.97, "No DOI values found for the chosen DOI type "
                        f"({PARAMS[prm]['doi'][doi_idx]}) – nothing to mask",
                        transform=ax.transAxes, ha='center', va='top', fontsize=8,
                        color='firebrick', zorder=6,
                        bbox=dict(facecolor='white', edgecolor='firebrick', alpha=0.85, pad=2))
            if mask_doi and okd.any():
                if interp and okd.sum() > 1:
                    xs = np.concatenate([[x_lo], dist[okd], [x_hi]])
                    ys = np.concatenate([[doi_y[okd][0]], doi_y[okd], [doi_y[okd][-1]]])
                    if 0 < mask_alpha < 0.999:
                        ax.fill_between(xs, ys, y_floor, color='white', alpha=mask_alpha, linewidth=0, zorder=3)
                    ax.plot(xs, ys, color='black', linestyle='--', linewidth=0.8, zorder=4)
                else:
                    ii = np.where(okd)[0]
                    l_, r_ = dist[ii] - bar_width / 2.0, dist[ii] + bar_width / 2.0
                    dy = doi_y[ii]
                    if 0 < mask_alpha < 0.999:
                        yb_ = lay['yb'][ii]
                        bottom = (np.nanmin(yb_, axis=1) if coord == 'elevation' else np.nanmax(yb_, axis=1))
                        bottom = np.where(np.isfinite(bottom), bottom, y_floor)
                        mverts = np.stack([np.stack([l_, dy], 1), np.stack([r_, dy], 1),
                                           np.stack([r_, bottom], 1), np.stack([l_, bottom], 1)], axis=1)
                        ax.add_collection(matplotlib.collections.PolyCollection(
                            mverts, facecolors='white', edgecolors='none', linewidths=0,
                            alpha=mask_alpha, zorder=3))
                    segs = np.stack([np.stack([l_, dy], 1), np.stack([r_, dy], 1)], axis=1)
                    ax.add_collection(matplotlib.collections.LineCollection(
                        segs, colors='black', linestyles='--', linewidths=0.8, zorder=4))

            if bhs:
                lith_used.update(draw_boreholes(ax, bhs, bh_w, bh_y, labels=(k == 0)))
                if wl_on:
                    n_wl = draw_water_levels(ax, bhs, bh_w, wl_y)
            mappable = artist
            if mappable is None:
                mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
                mappable.set_array([])
            cbar = horizontal_colorbar(self.figure, gs, k, mappable, PARAMS[prm]['cb'])
            if coord == 'elevation':
                ax.set_ylabel("Elevation (m)")
                ax.set_ylim(view_lo, view_hi)
            else:
                ax.set_ylabel("Depth (m)")
                ax.set_ylim(view_hi, view_lo)
            ax.grid(True, linestyle='--', alpha=0.3)
            if k < n_p - 1:
                ax.tick_params(labelbottom=False)
            self._panels[prm] = dict(ax=ax, artist=artist, cbar=cbar, data=lay['vals'].ravel(), lay=lay)

        self._plot_dist = dist
        self._plot_xy = rows[['X', 'Y']].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
        self.ax = self._axes[0]
        self._color_artist = self._panels[params[0]]['artist']
        self._cbar = self._panels[params[0]]['cbar']
        self._rho_data = self._panels.get('RHO', self._panels[params[0]])['data']

        # data misfit on a secondary (right) axis of the top panel
        if show_misfit:
            okm = np.isfinite(misfit)
            self.ax2 = self.ax.twinx()
            self.ax2.plot(dist[okm], misfit[okm], color='black', linewidth=1.2, marker='o',
                          markersize=3, zorder=6)
            self.ax2.axhline(1.0, color='grey', linestyle=':', linewidth=0.8)
            self.ax2.set_ylabel(f"Data misfit ({misfit_col})")
            self.ax2.set_ylim(0, self.misfit_max_spin.value())

        bottom_ax = self._axes[-1]
        if bhs:
            self._bh_drawn = (bhs, bh_w, bh_y)
            if lith_used or wl_on:
                handles = [mpatches.Patch(facecolor=c, edgecolor='black', linewidth=0.4, label=n)
                           for n, c in list(lith_used.items())[:15]]
                if wl_on:
                    handles.append(WL_HANDLE())
                if topo_on:
                    handles.append(self._topo_handle())
                leg = bottom_ax.legend(handles=handles, loc=getattr(self, '_bh_leg_pos', None) or 'lower right',
                                       fontsize=7, framealpha=0.85, ncol=min(3, (len(handles) + 4) // 5),
                                       title="Boreholes", title_fontsize=7)
                leg.set_draggable(True)          # the user can move the borehole legend
                self._bh_leg = leg
                if not getattr(self, '_bh_leg_cid', None):
                    self._bh_leg_cid = self.canvas.mpl_connect('button_release_event', self._remember_legend_pos)
        if topo_on and not (bhs and (lith_used or wl_on)):
            self.ax.legend(handles=[self._topo_handle()], loc='upper right', fontsize=7, framealpha=0.85)
        start_lab, end_lab = self._direction_labels()
        if self._single:
            r = rows.iloc[0]
            for ax in self._axes:
                ax.set_xticks([])
            self.ax.text(0.0, 1.01, f"Sounding {int(r['_SRC_RECORD'])} ({r['_SOURCE']})",
                         transform=self.ax.transAxes, ha='left', va='bottom', fontsize=10)
        else:
            bottom_ax.set_xlabel("Distance (m)")
            for xpos, lab, ha in ((0.0, start_lab, 'left'), (1.0, end_lab, 'right')):
                self.ax.text(xpos, 1.01, lab, transform=self.ax.transAxes, ha=ha, va='bottom',
                             fontsize=12, fontweight='bold')
        self.figure.tight_layout()
        self._home_lims = {a_: (a_.get_xlim(), a_.get_ylim()) for a_ in
                           list(self._axes) + ([self.ax2] if self.ax2 is not None else [])}
        self.canvas.draw_idle()

        # export table: every layer, all plotted parameters
        yt_all, yb_all = to_y(top_d), to_y(bot_d)
        rdoi = doi_of('RHO')
        for i in range(ns):
            for j in range(L):
                rec = {'Distance': dist[i], 'Top': yt_all[i, j], 'Bottom': yb_all[i, j],
                       'Sounding_ID': prof_df[id_col].iloc[i], 'Layer': j + 1,
                       'DOI': rdoi[i], 'Misfit': misfit[i]}
                for prm in params:
                    rec[PARAMS[prm]['export']] = layers_full[prm][i, j]
                rows_for_table.append(rec)
        self._last_plot_data = pd.DataFrame(rows_for_table)

        self._line_geom = line_geom
        self._has_plot = True
        what = " + ".join(PARAMS[p_]['label'] for p_ in params)
        what = f" ({what})" if len(self._params_avail) > 1 else ""
        if bar_px:
            what += f", bars {bar_value:g} px = {bar_width:.0f} m"
        if self.topo_check.isChecked():
            what += f", {topo_msg}" if topo_msg else ", topography shown"
            self.topo_info.setText(topo_msg[:1].upper() + topo_msg[1:] + "." if topo_msg else "")
        if bhs:
            what += f", {len(bhs)} borehole{'s' if len(bhs) > 1 else ''}"
            if self.wl_check.isChecked():
                n_wl = sum(1 for b in bhs if np.isfinite(b.get('wl', np.nan)))
                what += f" ({n_wl} with water level)"
                if wl_msg:
                    what += f" – {wl_msg}"
        elif getattr(self, 'bh_check', None) is not None and self.bh_check.isChecked() and not self._single:
            what += f", no boreholes within {self.bh_dist_spin.value():g} m"
        self.status_label.setText(
            f"Single sounding plotted{what}." if self._single else
            f"Profile {start_lab} → {end_lab}: {n_snd} soundings, {line_geom.length():.0f} m long{what}.")

    # -----------------------------------------------------------------
    #  Topography from a raster layer
    # -----------------------------------------------------------------
    def _refresh_topo_layers(self, *args):
        from qgis.core import QgsRasterLayer
        cur = self.topo_combo.currentData()
        rasters = [l for l in QgsProject.instance().mapLayers().values()
                   if isinstance(l, QgsRasterLayer) and l.isValid()]
        rasters.sort(key=lambda l: (not self._is_elevation_raster(l), l.name().lower()))
        self.topo_combo.blockSignals(True)
        self.topo_combo.clear()
        for l in rasters:
            self.topo_combo.addItem(l.name() + ("" if self._is_elevation_raster(l) else "  (image?)"), l.id())
        if not rasters:
            self.topo_combo.addItem("(no raster layers)", None)
        k = self.topo_combo.findData(cur) if cur else -1
        self.topo_combo.setCurrentIndex(k if k >= 0 else 0)
        self.topo_combo.blockSignals(False)
        self.topo_check.setEnabled(bool(rasters))
        self._topo_layer_changed(replot=False)

    @staticmethod
    def _is_elevation_raster(lyr):
        """A single-band raster from a file/database (DEM) – not a web map or an RGB image."""
        try:
            if lyr.providerType() in ('wms', 'xyz', 'arcgismapserver', 'vectortile'):
                return False
            return lyr.bandCount() == 1
        except Exception:
            return False

    def _topo_layer_changed(self, *args, replot=True):
        lyr = QgsProject.instance().mapLayer(self.topo_combo.currentData() or '')
        if lyr is None:
            self.topo_info.setText("")
        elif not self._is_elevation_raster(lyr):
            self.topo_info.setText("This layer looks like an image (web map / several bands), not an elevation "
                                   "model – choose a DEM to get a topography line.")
        else:
            self.topo_info.setText("")
        if replot and self.topo_check.isChecked():
            self._replot_if_plotted()

    def _topo_handle(self):
        from matplotlib.lines import Line2D
        lyr = QgsProject.instance().mapLayer(self.topo_combo.currentData() or '')
        return Line2D([], [], color=TOPO_COLOR, lw=1.4,
                      label=f"Topography ({lyr.name()})" if lyr is not None else "Topography")

    def _sample_topography(self, line_geom):
        """(distance, elevation) arrays sampled from the chosen raster along the profile line,
        and a message ('' if fine). Cached per layer and line."""
        lyr = QgsProject.instance().mapLayer(self.topo_combo.currentData() or '')
        if lyr is None:
            return None, "choose a raster layer for the topography"
        length = float(line_geom.length())
        if length <= 0:
            return None, ""
        key = (lyr.id(), round(length, 3), tuple((round(p.x(), 2), round(p.y(), 2)) for p in self.profile_points))
        if key in self._topo_cache:
            return self._topo_cache[key]
        data_crs = self.data_crs if (self.data_crs is not None and self.data_crs.isValid()) \
            else QgsProject.instance().crs()
        tr = None
        if lyr.crs().isValid() and data_crs.isValid() and lyr.crs() != data_crs:
            tr = QgsCoordinateTransform(data_crs, lyr.crs(), QgsProject.instance())
        # step: about one raster cell (in metres when the CRSs match), 200 - 2000 samples
        try:
            cell = abs(float(lyr.rasterUnitsPerPixelX())) if tr is None and not lyr.crs().isGeographic() else 0.0
        except Exception:
            cell = 0.0
        n = int(min(max(length / max(cell, length / 2000.0), 200), 2000))
        d = np.linspace(0.0, length, n + 1)
        prov = lyr.dataProvider()
        z = np.full(len(d), np.nan)
        for i, di in enumerate(d):
            g = line_geom.interpolate(float(di))
            if g is None or g.isEmpty():
                continue
            pt = g.asPoint()
            if tr is not None:
                pt = tr.transform(pt)
            try:
                val, ok = prov.sample(QgsPointXY(pt), 1)
            except Exception:
                val, ok = None, False
            if ok and val is not None and np.isfinite(val):
                z[i] = float(val)
        msg = ""
        if not np.isfinite(z).any():
            res = (None, "the raster has no values along this line (outside the DEM, or not an elevation model)")
            self._topo_cache[key] = res
            return res
        if not self._is_elevation_raster(lyr):
            msg = f"'{lyr.name()}' is an image, so the line shows its pixel values, not heights"
        res = ((d, z), msg)
        self._topo_cache[key] = res
        return res

    # -----------------------------------------------------------------
    #  Bar width unit and water levels
    # -----------------------------------------------------------------
    def _bar_unit_changed(self, *args):
        new = self.barunit_combo.currentData()
        self._bw_vals[self._bw_unit] = self.barwidth_edit.text()
        self._bw_unit = new
        self.barwidth_edit.setText(self._bw_vals.get(new, "10"))
        self._replot_if_plotted()

    def _wl_index(self, key):
        for i in range(self.wl_combo.count()):
            if self.wl_combo.itemData(i) == key:
                return i
        return -1

    def _wl_tables(self, lyr):
        """{(source, column)-tables} of one borehole layer (cached)."""
        key = lyr.id()
        if key not in self._wl_cache:
            try:
                self._wl_cache[key] = wlv.borehole_tables(self.registry, lyr)
            except Exception as exc:
                _log(f"Water level tables of {lyr.name()}: {exc}")
                self._wl_cache[key] = {}
        return self._wl_cache[key]

    def _refresh_wl_columns(self, *args):
        """Fill the water-level column list from the ticked borehole layers."""
        if not self.wl_check.isChecked():
            return                      # built when the water level is switched on
        cur = self.wl_combo.currentData()
        alive = {l.id() for l in self.bh_picker.checked_layers()}
        self._wl_cache = {k: v for k, v in self._wl_cache.items() if k in alive}
        entries = {}
        for lyr in self.bh_picker.checked_layers():
            for src, tbl in self._wl_tables(lyr).items():
                for col in tbl['df'].columns:
                    if str(col).startswith('_') or col == tbl.get('name_col'):
                        continue
                    entries.setdefault(wlv.key_of(src, col), wlv.label_of(src, col))
        self.wl_combo.blockSignals(True)
        self.wl_combo.clear()
        items = sorted(entries.items(), key=lambda kv: (not wlv.looks_like_wl(kv[0]), kv[1].lower()))
        for k, lab in items:
            self.wl_combo.addItem(lab, k)
        if not items:
            self.wl_combo.addItem("(tick a borehole layer)", None)
        i = self._wl_index(cur) if cur else -1
        self.wl_combo.setCurrentIndex(i if i >= 0 else 0)
        self.wl_combo.blockSignals(False)
        self.wl_info.setText("" if items else "No borehole columns found.")

    def _water_levels(self, bhs):
        """{(layer id, borehole name): value} from the chosen column, and a short message."""
        key = self.wl_combo.currentData()
        out, matched, layers_wo = {}, 0, []
        proj = QgsProject.instance()
        for lid in {b.get('layer_id') for b in bhs}:
            lyr = proj.mapLayer(lid or '')
            if lyr is None:
                continue
            vals = wlv.values_for(self._wl_tables(lyr), key)
            if vals is None:
                layers_wo.append(lyr.name())
                continue
            for b in bhs:
                if b.get('layer_id') == lid:
                    v = vals.get(wlv.norm_name(b['name']))
                    if v is not None and np.isfinite(v):
                        out[(lid, b['name'])] = v
                        matched += 1
        msg = ""
        if layers_wo:
            msg = f"column not in {', '.join(layers_wo)}"
        self.wl_info.setText(f"{matched} of {len(bhs)} boreholes on the profile have a water level"
                             + (f" ({msg})" if msg else "") + ".")
        return out, msg

    # -----------------------------------------------------------------
    #  Model parameters (IP) and per-parameter colour scales
    # -----------------------------------------------------------------
    def _update_param_ui(self):
        avail = set()
        for lyr in self.snd_picker.checked_layers():
            try:
                avail.update(available_params(pd.DataFrame(columns=self.registry.sounding_columns(lyr))))
            except Exception:
                pass
        self._params_avail = [p for p in PARAM_ORDER if p in avail] or ['RHO']
        ip = len(self._params_avail) > 1
        for prm in self._params_avail:          # a checkbox for parameters found in this file
            if prm not in self.param_checks:
                cb = QCheckBox(PARAMS[prm]['label'])
                cb.setToolTip(f"{PARAMS[prm]['cb']} – tick several to stack them (top to bottom)")
                cb.toggled.connect(self._replot_if_plotted)
                lay = self.param_row.layout()
                lay.insertWidget(lay.count() - 1, cb)
                self.param_checks[prm] = cb
        self.param_row.setVisible(ip)
        for prm, cb in self.param_checks.items():
            cb.setVisible(prm in self._params_avail)
        self.cs_param_label.setVisible(ip)
        self.cs_param_combo.setVisible(ip)
        cur = self._cs_param
        self.cs_param_combo.blockSignals(True)
        self.cs_param_combo.clear()
        for prm in self._params_avail:
            self.cs_param_combo.addItem(PARAMS[prm]['label'], prm)
        k = self.cs_param_combo.findData(cur)
        self.cs_param_combo.setCurrentIndex(k if k >= 0 else 0)
        self.cs_param_combo.blockSignals(False)
        if self.cs_param_combo.currentData() != cur:
            self._on_cs_param_changed()

    def _selected_params(self):
        if len(self._params_avail) <= 1:
            return ['RHO']
        sel = [p for p in PARAM_ORDER if p in self._params_avail and self.param_checks[p].isChecked()]
        return sel or ['RHO']

    def _widget_cs(self):
        return dict(cmap=self.cmap_combo.currentText(), rev=self.cmap_reverse_check.isChecked(),
                    log=self.cscale_combo.currentText() == "Logarithmic",
                    vmin=self.cmin_spin.value(), vmax=self.cmax_spin.value())

    def _load_cs(self, prm):
        cs = self._cs.get(prm)
        if cs is None:
            return
        info = PARAMS[prm]
        widgets = (self.cmap_combo, self.cmap_reverse_check, self.cscale_combo, self.cmin_spin, self.cmax_spin)
        for w in widgets:
            w.blockSignals(True)
        self.cmap_combo.setCurrentText(cs['cmap'])
        self._last_cmap_name = cs['cmap']
        self.cmap_reverse_check.setChecked(cs['rev'])
        self.cscale_combo.setCurrentText("Logarithmic" if cs['log'] else "Linear")
        for sp in (self.cmin_spin, self.cmax_spin):
            sp.setDecimals(info['decimals'])
            sp.setSuffix(f" {info['unit']}" if info['unit'] else "")
        self.cmin_spin.setValue(cs['vmin'])
        self.cmax_spin.setValue(cs['vmax'])
        for w in widgets:
            w.blockSignals(False)

    def _on_cs_param_changed(self, *args):
        new = self.cs_param_combo.currentData() or 'RHO'
        self._cs[self._cs_param] = self._widget_cs()
        self._cs_param = new
        if new not in self._cs:
            vals = []
            for lyr in self.snd_picker.checked_layers():
                try:
                    vals.append(layer_values(self.registry.soundings_df(lyr), new).ravel())
                except Exception:
                    pass
            self._cs[new] = self._default_cs(new, np.concatenate(vals) if vals else None)
        self._load_cs(new)

    def _default_cs(self, prm, values):
        """First colour scale of a parameter: RES keeps the classic default, IP
        parameters get BlueToPink with a range from the data (C linear, others log)."""
        if prm == 'RHO':
            return dict(cmap='Spectral', rev=True, log=True, vmin=1.0, vmax=1000.0)
        v = np.asarray(values if values is not None else [], dtype=float)
        v = v[np.isfinite(v)]
        if prm in ('PHI', 'TPHI', 'TAU'):
            log = True
        elif prm in ('C', 'M0'):
            log = False
        else:                       # other parameters: log when positive and spanning > 2 decades
            log = bool(v.size) and v.min() > 0 and v.max() / v.min() > 100
        lo, hi = (1.0, 1000.0) if log else (0.0, 1.0)
        v = v[(v > 0) if log else np.ones(len(v), bool)]
        if v.size:
            lo, hi = np.percentile(v, [2, 98])
            if hi <= lo:
                hi = lo * 10 if log else lo + 1
        return dict(cmap='BlueToPink', rev=False, log=log, vmin=float(f"{lo:.3g}"), vmax=float(f"{hi:.3g}"))

    def _cs_for(self, prm):
        if prm == self._cs_param:
            return self._widget_cs()
        return self._cs.get(prm)

    def _cmap_norm_for(self, prm, values=None):
        cs = self._cs_for(prm)
        if cs is None:
            cs = self._cs[prm] = self._default_cs(prm, values)
        cmap = get_colormap(cs['cmap'], cs['rev'])
        if cmap is None:
            return None, None, f"Unknown colormap '{cs['cmap']}'."
        if cs['vmin'] >= cs['vmax']:
            return None, None, f"{PARAMS[prm]['label']} colour scale: Min must be smaller than Max."
        if cs['log']:
            if cs['vmin'] <= 0:
                return None, None, f"{PARAMS[prm]['label']} colour scale: a log scale needs Min > 0."
            return cmap, matplotlib.colors.LogNorm(vmin=cs['vmin'], vmax=cs['vmax']), None
        return cmap, matplotlib.colors.Normalize(vmin=cs['vmin'], vmax=cs['vmax']), None

    def _on_cmap_name_changed(self, name):
        """BlueToPink is already low=blue → high=pink, Spectral needs reversing; flip the
        Reverse box when switching between the two so both look as expected."""
        prev = getattr(self, '_last_cmap_name', '')
        self._last_cmap_name = name
        if name.lower() == 'bluetopink' and prev.lower() != 'bluetopink':
            self.cmap_reverse_check.setChecked(False)
        elif prev.lower() == 'bluetopink' and name.lower() == 'spectral':
            self.cmap_reverse_check.setChecked(True)

    def _current_cmap_norm(self):
        """(cmap, norm, error) of the parameter shown in the Color scale box."""
        return self._cmap_norm_for(self._cs_param)

    def _on_scale_type_changed(self, *args):
        self._color_timer.start()

    def _apply_color_scale(self):
        """Re-colour the panel of the current parameter without recomputing the profile."""
        prm = self._cs_param
        cmap, norm, err = self._cmap_norm_for(prm)
        if err:
            self.status_label.setText(err)
            return
        self._cs[prm] = self._widget_cs()
        panel = self._panels.get(prm)
        if panel is None or panel['artist'] is None:
            return   # not plotted – used on the next plot
        art = panel['artist']
        art.set_cmap(cmap)
        art.set_norm(norm)
        if panel['cbar'] is not None:
            try:
                panel['cbar'].update_normal(art)
            except Exception:
                pass
        self.canvas.draw_idle()
        unit = PARAMS[prm]['unit']
        self.status_label.setText(
            f"{PARAMS[prm]['label']} colour scale: {cmap.name}, {self.cscale_combo.currentText().lower()}, "
            f"{norm.vmin:g}–{norm.vmax:g} {unit}")

    def _auto_color_range(self):
        panel = self._panels.get(self._cs_param)
        if panel is None:
            QMessageBox.information(self, "Color scale", "Plot this parameter first.")
            return
        data = np.asarray(panel['data'], dtype=float)
        log = self.cscale_combo.currentText() == "Logarithmic"
        data = data[np.isfinite(data) & ((data > 0) if log else True)]
        if data.size == 0:
            return
        lo, hi = np.percentile(data, [2, 98])
        if hi <= lo:
            lo, hi = data.min(), data.max()
        if hi <= lo:
            hi = lo * 10 if log and lo > 0 else lo + 1
        lo, hi = float(f"{lo:.2g}"), float(f"{hi:.2g}")
        if hi <= lo:
            hi = lo * 1.1 if lo > 0 else lo + 1
        for sp, v in ((self.cmin_spin, lo), (self.cmax_spin, hi)):
            sp.blockSignals(True)
            sp.setValue(v)
            sp.blockSignals(False)
        self._apply_color_scale()

    def _canvas_transform(self, to_canvas=True):
        canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
        data_crs = self.data_crs if (self.data_crs and self.data_crs.isValid()) else canvas_crs
        src, dst = (data_crs, canvas_crs) if to_canvas else (canvas_crs, data_crs)
        return QgsCoordinateTransform(src, dst, QgsProject.instance())

class ResistivityToolDialog(QDialog):
    """Stand-alone window of the TEM-Viewer Profile Tool."""

    def __init__(self, parent=None, iface=None):
        super().__init__(parent)
        self.setWindowTitle("TEM-Viewer - Profile Tool")
        self._torn_down = False
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowMinimizeButtonHint |
                            Qt.WindowType.WindowMaximizeButtonHint | Qt.WindowType.WindowCloseButtonHint)
        self.resize(1250, 800)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.widget = ResistivityProfileWidget(self, iface=iface, embedded=False)
        lay.addWidget(self.widget)

    def closeEvent(self, event):
        self._teardown()
        super().closeEvent(event)

    def done(self, result):
        # closing the window ends the session; the next start is fresh
        self._teardown()
        super().done(result)

    def _teardown(self):
        if self._torn_down:
            return
        self._torn_down = True
        self.widget.cleanup()
        try:
            self.widget.registry.disconnect_project()
        except Exception:
            pass
