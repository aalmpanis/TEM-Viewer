# -*- coding: utf-8 -*-
"""
TEM-Viewer - Geological Model tab.

Pick a profile line (e.g. one saved from the Profile tab), show the resistivity
(or IP) section along it as a background, digitise the surfaces that separate the
geological formations, name the formations and plot the geological section on
its own. Surfaces can be exported to the map as 3D lines and the interpretation
can be saved to / loaded from a .json file.
"""
import json
import math

import numpy as np
import pandas as pd

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel, QPushButton,
    QComboBox, QCheckBox, QDoubleSpinBox, QSplitter, QScrollArea, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QMessageBox, QFileDialog, QColorDialog, QHeaderView,
    QInputDialog, QApplication
)
from qgis.core import (QgsProject, QgsVectorLayer, QgsGeometry, QgsPointXY, QgsFeature,
                       QgsField, QgsCoordinateTransform, Qgis)
from qgis.gui import QgsVertexMarker

from .tv_compat import (FigureCanvasQTAgg as FigureCanvas,
                             NavigationToolbar2QT as NavigationToolbar, qgis_enum,
                             FIELD_STRING, FIELD_DOUBLE)
import matplotlib
import matplotlib.colors
import matplotlib.collections
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

from .tv_data import LayerPicker
from .borehole_section import (project_on_line, boreholes_near_line, draw_boreholes, lith_at,
                               lith_rgb_or_none, draw_sounding_frames)
from .resistivity_tool import (PARAMS, PARAM_ORDER, available_params, layer_geometry, layer_values,
                               get_colormap, RESISTIVITY_CMAPS, PROFILE_KIND)

FORMATION_COLORS = ['#f2d16b', '#c8a165', '#8fbf6a', '#6aa5d8', '#b07cc6', '#e58b6d',
                    '#9e9e9e', '#d4e157', '#4db6ac', '#f48fb1']
MARGIN = 20.0          # m above the highest ground / below the deepest DOI
GEO_KIND = 'temviewer_geomodels'     # custom property of the saved-models layer


# =============================================================================
#  Geometry helpers (no Qt)
# =============================================================================
def point_at(line, s):
    """XY of the point at distance s along the polyline (s may be an array)."""
    seg = np.diff(line, axis=0)
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
    s = np.clip(np.atleast_1d(np.asarray(s, dtype=float)), 0.0, cum[-1])
    return np.column_stack([np.interp(s, cum, line[:, 0]), np.interp(s, cum, line[:, 1])])


def line_length(line):
    seg = np.diff(line, axis=0)
    return float(np.sum(np.hypot(seg[:, 0], seg[:, 1])))


def compute_section(df, line, tolerance, param='RHO', doi_kind=0):
    """Soundings within `tolerance` of the line, sorted along it, with layer geometry."""
    xy = df[['X', 'Y']].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    ok = np.isfinite(xy).all(axis=1)
    s, off = project_on_line(xy[ok], line)
    idx = np.where(ok)[0][off <= tolerance]
    s = s[off <= tolerance]
    if not len(idx):
        return None
    order = np.argsort(s, kind='stable')
    rows = df.iloc[idx[order]]
    dist = s[order]
    top, bot = layer_geometry(rows)
    L = top.shape[1]
    vals = layer_values(rows, param)[:, :L] if param else np.full(top.shape, np.nan)
    elev = pd.to_numeric(rows['ELEVATION'], errors='coerce').to_numpy(dtype=float)
    doi = np.full(len(rows), np.nan)
    pref = PARAMS[param]['doi'][doi_kind] if param else None
    alt = PARAMS['RHO']['doi'][doi_kind]
    col = pref if pref in rows.columns else (alt if alt in rows.columns else None)
    if col is not None:
        d = pd.to_numeric(rows[col], errors='coerce').to_numpy(dtype=float)
        doi = np.where(d > 0, d, np.nan)
    return dict(dist=dist, elev=elev, top=top, bot=bot, vals=vals, doi=doi,
                ids=rows['RECORD'].to_numpy(), n=len(rows),
                xy=rows[['X', 'Y']].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float))


def surface_curve(pts, x):
    """Elevation of a picked surface at distances x (linear, flat beyond the ends)."""
    if not pts:
        return np.full(len(x), np.nan)
    p = np.asarray(sorted(pts), dtype=float)
    return np.interp(x, p[:, 0], p[:, 1])


def ordered_boundaries(surfaces, x, ground, bottom):
    """Surfaces sorted top → bottom (by mean elevation), each clipped so it never
    rises above the ground or the surface above it. Returns [(surface, z), ...]."""
    curves = [(s, surface_curve(s['pts'], x)) for s in surfaces if s['pts']]
    curves.sort(key=lambda sc: -float(np.nanmean(sc[1])))
    out, upper = [], ground.copy()
    for s, z in curves:
        z = np.minimum(np.maximum(z, bottom), upper)
        out.append((s, z))
        upper = z
    return out


# =============================================================================
#  Tab
# =============================================================================
class GeoModelWidget(QWidget):
    def __init__(self, parent=None, iface=None, registry=None):
        super().__init__(parent)
        self.iface = iface
        self.registry = registry
        self.sec = None              # section along the profile
        self.line = None             # profile polyline (n x 2) in the sounding CRS
        self.data_crs = None
        self.prof_name = ""
        self.top_unit = dict(label="Formation 1", color=FORMATION_COLORS[0])
        self.surfaces = []           # dicts: name, label (formation below), color, pts [(d, z)]
        self.view = 'edit'           # 'edit' (section + surfaces) or 'geology'
        self.ax = None
        self._art = None
        self.pos_marker = None
        self._bg_param = None
        self._setup_ui()
        if iface is not None:
            self.pos_marker = QgsVertexMarker(iface.mapCanvas())
            self.pos_marker.setIconType(qgis_enum(QgsVertexMarker, 'IconType', 'ICON_CROSS'))
            self.pos_marker.setColor(Qt.GlobalColor.blue)
            self.pos_marker.setIconSize(18)
            self.pos_marker.setPenWidth(3)
            self.pos_marker.hide()

    # ------------------------------------------------------------------
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        split = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(split, 1)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(9, 5))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(400, 300)
        self.toolbar = NavigationToolbar(self.canvas, self)
        lv.addWidget(self.toolbar)
        lv.addWidget(self.canvas, 1)
        self.cursor_label = QLabel("Distance:   Elevation:")
        self.cursor_label.setStyleSheet("font-family: monospace; font-size: 10px;")
        lv.addWidget(self.cursor_label)
        bh = QHBoxLayout()
        self.edit_btn = QPushButton("Edit Surfaces")
        self.edit_btn.setToolTip("Section with the picked surfaces (picking mode)")
        self.edit_btn.clicked.connect(lambda: self._set_view('edit'))
        self.geo_btn = QPushButton("Plot Geological Formations")
        self.geo_btn.setStyleSheet("font-weight: bold;")
        self.geo_btn.setToolTip("Plot only the geological formations between the surfaces")
        self.geo_btn.clicked.connect(lambda: self._set_view('geology'))
        self.reset_btn = QPushButton("Reset view")
        self.reset_btn.clicked.connect(self._reset_view)
        self.export_btn = QPushButton("Surfaces to Map")
        self.export_btn.setToolTip("Add the surfaces as 3D lines (Z = elevation) to the map")
        self.export_btn.clicked.connect(self._export_surfaces)
        self.save_btn = QPushButton("Save…")
        self.save_btn.setToolTip("Save the interpretation (surfaces, formations) to a .json file")
        self.save_btn.clicked.connect(self._save_json)
        self.load_btn = QPushButton("Load…")
        self.load_btn.setToolTip("Load an interpretation saved before")
        self.load_btn.clicked.connect(self._load_json)
        for w in (self.edit_btn, self.geo_btn, self.reset_btn):
            bh.addWidget(w)
        bh.addStretch()
        self.png_btn = QPushButton("Save Image…")
        self.png_btn.setToolTip("Save the section as shown (edit or geological view) as PNG, PDF or SVG")
        self.png_btn.clicked.connect(self._save_image)
        self.model_btn = QPushButton("Save Model as Layer")
        self.model_btn.setToolTip("Store the geological section (with its location map) in the 'TEM-Viewer geological "
                                  "models' layer")
        self.model_btn.clicked.connect(self._save_model_layer)
        for w in (self.png_btn, self.model_btn, self.export_btn, self.save_btn, self.load_btn):
            bh.addWidget(w)
        lv.addLayout(bh)
        split.addWidget(left)

        panel = QWidget()
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(0, 0, 4, 0)

        g1 = QGroupBox("1. Profile")
        s1 = QGridLayout(g1)
        self.line_combo = QComboBox()
        self.line_combo.setToolTip("Line layer with the profile, e.g. 'TEM-Viewer profiles' saved from the Profile tab")
        self.line_combo.currentIndexChanged.connect(self._fill_features)
        self.line_refresh = QPushButton("↻")
        self.line_refresh.setMaximumWidth(28)
        self.line_refresh.clicked.connect(self._refresh_lines)
        self.feat_combo = QComboBox()
        self.feat_combo.currentIndexChanged.connect(self._feature_settings)
        self.picker = LayerPicker(self.registry, 'soundings', self, max_height=70)
        self.picker.selectionChangedByUser.connect(self._update_params)
        self.tol_spin = QDoubleSpinBox()
        self.tol_spin.setRange(0, 1e6)
        self.tol_spin.setDecimals(0)
        self.tol_spin.setValue(50)
        self.tol_spin.setSuffix(" m")
        self.tol_spin.setToolTip("Soundings up to this distance from the line are used")
        self.load_sec_btn = QPushButton("Load Section")
        self.load_sec_btn.setStyleSheet("font-weight: bold;")
        self.load_sec_btn.clicked.connect(self.load_section)
        s1.addWidget(QLabel("Line layer:"), 0, 0)
        s1.addWidget(self.line_combo, 0, 1)
        s1.addWidget(self.line_refresh, 0, 2)
        s1.addWidget(QLabel("Profile:"), 1, 0)
        s1.addWidget(self.feat_combo, 1, 1, 1, 2)
        s1.addWidget(QLabel("Sounding layers:"), 2, 0, 1, 3)
        s1.addWidget(self.picker, 3, 0, 1, 3)
        s1.addWidget(QLabel("Search width:"), 4, 0)
        s1.addWidget(self.tol_spin, 4, 1)
        s1.addWidget(self.load_sec_btn, 5, 0, 1, 3)
        pv.addWidget(g1)

        g2 = QGroupBox("2. Background")
        s2 = QGridLayout(g2)
        self.bg_combo = QComboBox()
        self.bg_combo.addItem("RES", 'RHO')
        self.bg_combo.addItem("None", None)
        self.bg_combo.currentIndexChanged.connect(self._bg_changed)
        self.bg_cmap = QComboBox()
        self.bg_cmap.setEditable(True)
        self.bg_cmap.addItems([c for c in RESISTIVITY_CMAPS if get_colormap(c) is not None])
        self.bg_cmap.setCurrentText('BlueToPink')
        self.bg_min, self.bg_max = QDoubleSpinBox(), QDoubleSpinBox()
        for sp, v in ((self.bg_min, 1.0), (self.bg_max, 1000.0)):
            sp.setRange(0, 1e9)
            sp.setDecimals(3)
            sp.setValue(v)
            sp.setKeyboardTracking(False)
        self.bg_alpha = QDoubleSpinBox()
        self.bg_alpha.setRange(0, 100)
        self.bg_alpha.setDecimals(0)
        self.bg_alpha.setValue(100)
        self.bg_alpha.setSuffix(" %")
        self.bg_alpha.setToolTip("Opacity of the background section")
        self.doi_check = QCheckBox("Mask below DOI")
        self.doi_check.setChecked(True)
        self.doi_alpha = QDoubleSpinBox()
        self.doi_alpha.setRange(0, 100)
        self.doi_alpha.setDecimals(0)
        self.doi_alpha.setSingleStep(10)
        self.doi_alpha.setValue(60)
        self.doi_alpha.setSuffix(" %")
        self.doi_alpha.setKeyboardTracking(False)
        self.doi_alpha.setToolTip("How strongly the section below the DOI is covered with white: "
                                  "0 % = not at all, 100 % = hidden completely")
        self.doi_check.toggled.connect(self.doi_alpha.setEnabled)
        s2.addWidget(QLabel("Show:"), 0, 0)
        s2.addWidget(self.bg_combo, 0, 1)
        s2.addWidget(self.bg_cmap, 0, 2)
        s2.addWidget(QLabel("Min / Max:"), 1, 0)
        s2.addWidget(self.bg_min, 1, 1)
        s2.addWidget(self.bg_max, 1, 2)
        s2.addWidget(QLabel("Section opacity:"), 2, 0)
        s2.addWidget(self.bg_alpha, 2, 1)
        s2.addWidget(self.doi_check, 3, 0)
        s2.addWidget(QLabel("Mask strength:"), 3, 1)
        s2.addWidget(self.doi_alpha, 3, 2)
        self.bh_check = QCheckBox("Boreholes within")
        self.bh_check.setToolTip("Draw the boreholes of the ticked borehole layers that lie within this "
                                 "distance of the profile line, on top of the section")
        self.bh_dist_spin = QDoubleSpinBox()
        self.bh_dist_spin.setRange(0, 1e6)
        self.bh_dist_spin.setDecimals(0)
        self.bh_dist_spin.setValue(100)
        self.bh_dist_spin.setSuffix(" m")
        self.bh_dist_spin.setKeyboardTracking(False)
        self.bh_picker = LayerPicker(self.registry, 'boreholes', self, max_height=55)
        self.bh_picker.setVisible(False)
        s2.addWidget(self.bh_check, 3, 0, 1, 2)
        s2.addWidget(self.bh_dist_spin, 3, 2)
        s2.addWidget(self.bh_picker, 4, 0, 1, 3)
        self.bar_spin = QDoubleSpinBox()
        self.bar_spin.setRange(0, 1e5)
        self.bar_spin.setDecimals(0)
        self.bar_spin.setValue(0)
        self.bar_spin.setSuffix(" m")
        self.bar_spin.setSpecialValueText("auto")
        self.bar_spin.setKeyboardTracking(False)
        self.bar_spin.setToolTip("Width of the sounding columns; 'auto' fills half-way to the neighbours")
        self.interp_check = QCheckBox("Interpolate")
        self.interp_check.setToolTip("Interpolate between neighbouring soundings (not across large gaps)")
        s2.addWidget(QLabel("Bar width:"), 5, 0)
        s2.addWidget(self.bar_spin, 5, 1)
        s2.addWidget(self.interp_check, 5, 2)
        self.frames_check = QCheckBox("Sounding frames")
        self.frames_check.setToolTip("Outline each sounding (bar width) and its layers")
        s2.addWidget(self.frames_check, 6, 2)
        self.frames_check.toggled.connect(lambda *_: self._redraw())
        self.bh_width_spin = QDoubleSpinBox()
        self.bh_width_spin.setRange(0, 1e5)
        self.bh_width_spin.setDecimals(0)
        self.bh_width_spin.setValue(0)
        self.bh_width_spin.setSuffix(" m")
        self.bh_width_spin.setSpecialValueText("auto")
        self.bh_width_spin.setKeyboardTracking(False)
        self.bh_width_spin.setToolTip("Width of the borehole columns; 'auto' = 1.2 % of the profile length")
        self.bh_width_lbl = QLabel("Borehole width:")
        s2.addWidget(self.bh_width_lbl, 6, 0)
        s2.addWidget(self.bh_width_spin, 6, 1)
        for w_ in (self.bh_width_lbl, self.bh_width_spin):
            w_.setVisible(False)
            self.bh_check.toggled.connect(w_.setVisible)
        self.bh_width_spin.valueChanged.connect(lambda *_: self.bh_check.isChecked() and self._redraw())
        self.bar_spin.valueChanged.connect(lambda *_: self._redraw())
        self.interp_check.toggled.connect(lambda *_: self._redraw())
        self.bh_check.toggled.connect(self.bh_picker.setVisible)
        self.bh_check.toggled.connect(lambda *_: self._redraw())
        self.bh_dist_spin.valueChanged.connect(lambda *_: self.bh_check.isChecked() and self._redraw())
        self.bh_picker.selectionChangedByUser.connect(lambda: self.bh_check.isChecked() and self._redraw())
        for sig in (self.bg_cmap.currentTextChanged, self.bg_min.valueChanged, self.bg_max.valueChanged,
                    self.bg_alpha.valueChanged, self.doi_check.toggled, self.doi_alpha.valueChanged):
            sig.connect(lambda *_: self._redraw())
        pv.addWidget(g2)

        g3 = QGroupBox("3. Surfaces and formations (top → bottom)")
        s3 = QVBoxLayout(g3)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Surface", "Formation below", "Colour", "Points"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(150)
        self.table.itemChanged.connect(self._table_edited)
        self.table.cellDoubleClicked.connect(self._cell_double_clicked)
        self.table.itemSelectionChanged.connect(lambda: self._redraw())
        s3.addWidget(self.table)
        r1 = QHBoxLayout()
        self.add_btn = QPushButton("Add Surface")
        self.add_btn.clicked.connect(self._add_surface)
        self.del_btn = QPushButton("Remove")
        self.del_btn.clicked.connect(self._remove_surface)
        self.clear_btn = QPushButton("Clear Points")
        self.clear_btn.clicked.connect(self._clear_points)
        for w in (self.add_btn, self.del_btn, self.clear_btn):
            r1.addWidget(w)
        s3.addLayout(r1)
        self.pick_btn = QPushButton("Pick Points on Section")
        self.pick_btn.setCheckable(True)
        self.pick_btn.setToolTip("Left-click adds a point to the selected surface, "
                                 "right-click removes the nearest point")
        self.pick_btn.toggled.connect(self._pick_toggled)
        s3.addWidget(self.pick_btn)
        hint = QLabel("Select a surface, press Pick and click along the boundary on the section "
                      "(left-click = add, right-click = remove). Double-click a name to edit it, "
                      "the colour cell to change the colour. The first row is the ground surface.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555;")
        s3.addWidget(hint)
        pv.addWidget(g3)

        g4 = QGroupBox("4. Geological section")
        s4 = QGridLayout(g4)
        self.labels_check = QCheckBox("Formation labels")
        self.labels_check.setChecked(True)
        self.legend_check = QCheckBox("Legend")
        self.legend_check.setChecked(True)
        self.snd_check = QCheckBox("Sounding positions")
        self.snd_check.setChecked(True)
        self.surf_lines_check = QCheckBox("Surface lines")
        self.surf_lines_check.setChecked(True)
        for k, w in enumerate((self.labels_check, self.legend_check, self.snd_check, self.surf_lines_check)):
            s4.addWidget(w, k // 2, k % 2)
            w.toggled.connect(lambda *_: self._redraw())
        self.geo_alpha = QDoubleSpinBox()
        self.geo_alpha.setRange(0, 100)
        self.geo_alpha.setDecimals(0)
        self.geo_alpha.setSingleStep(10)
        self.geo_alpha.setValue(100)
        self.geo_alpha.setSuffix(" %")
        self.geo_alpha.setKeyboardTracking(False)
        self.geo_alpha.setToolTip("Opacity of the formations; below 100 % the background section "
                                  "(resistivity / IP, see 2.) shows through")
        self.geo_alpha.valueChanged.connect(lambda *_: self._redraw())
        s4.addWidget(QLabel("Formation opacity:"), 2, 0)
        s4.addWidget(self.geo_alpha, 2, 1)
        auto_lbl = QLabel("Formation colours follow their names (clay blue → sand yellow/red → gravel "
                          "dark red → bedrock purple) until you pick a colour yourself.")
        auto_lbl.setWordWrap(True)
        auto_lbl.setStyleSheet("color: #555;")
        s4.addWidget(auto_lbl, 3, 0, 1, 2)
        pv.addWidget(g4)
        pv.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(340)
        split.addWidget(scroll)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)

        self.status_label = QLabel("Choose a profile line and sounding layers, then Load Section.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.canvas.mpl_connect('motion_notify_event', self._on_move)
        self.canvas.mpl_connect('axes_leave_event', lambda e: self._hide_marker())
        self._refresh_lines()
        self._fill_table()
        self._update_params()

    # ------------------------------------------------------------------
    #  Profile line and settings
    # ------------------------------------------------------------------
    def _refresh_lines(self):
        cur = self.line_combo.currentData()
        self.line_combo.blockSignals(True)
        self.line_combo.clear()
        layers = [l for l in QgsProject.instance().mapLayers().values()
                  if isinstance(l, QgsVectorLayer) and l.geometryType() == Qgis.GeometryType.Line]
        layers.sort(key=lambda l: (l.customProperty("temviewer/kind") != PROFILE_KIND, l.name().lower()))
        for l in layers:
            self.line_combo.addItem(l.name(), l.id())
        k = self.line_combo.findData(cur) if cur else -1
        self.line_combo.setCurrentIndex(k if k >= 0 else (0 if layers else -1))
        self.line_combo.blockSignals(False)
        self._fill_features()

    def _line_layer(self):
        return QgsProject.instance().mapLayer(self.line_combo.currentData() or '')

    def _fill_features(self, *args):
        lyr = self._line_layer()
        self.feat_combo.blockSignals(True)
        self.feat_combo.clear()
        if lyr is not None:
            names = lyr.fields().names()
            sel = set(lyr.selectedFeatureIds())
            first_sel = -1
            for f in lyr.getFeatures():
                lab = str(f['name']) if 'name' in names and f['name'] not in (None, '') else f"Feature {f.id()}"
                self.feat_combo.addItem(lab, f.id())
                if f.id() in sel and first_sel < 0:
                    first_sel = self.feat_combo.count() - 1
            if first_sel >= 0:
                self.feat_combo.setCurrentIndex(first_sel)
        self.feat_combo.blockSignals(False)
        self._feature_settings()

    def _feature(self):
        lyr = self._line_layer()
        fid = self.feat_combo.currentData()
        if lyr is None or fid is None:
            return lyr, None
        for f in lyr.getFeatures():
            if f.id() == fid:
                return lyr, f
        return lyr, None

    def _feature_settings(self, *args):
        """A line saved by the Profile tab remembers its search width and sounding layers."""
        lyr, f = self._feature()
        if f is None:
            return
        key = (lyr.id(), f.id())
        if key == getattr(self, '_feat_key', None):
            return                      # same profile: keep the user's settings
        self._feat_key = key
        names = lyr.fields().names()
        try:
            if 'search_w' in names and f['search_w'] not in (None, ''):
                self.tol_spin.setValue(float(f['search_w']))
            if 'layer_ids' in names and f['layer_ids']:
                ids = [i for i in str(f['layer_ids']).split(';') if QgsProject.instance().mapLayer(i)]
                if ids:
                    self.picker.set_checked_ids(ids)
        except Exception:
            pass

    def _update_params(self, *args):
        avail = set()
        for lyr in self.picker.checked_layers():
            try:
                avail.update(available_params(pd.DataFrame(columns=self.registry.sounding_columns(lyr))))
            except Exception:
                pass
        params = [p for p in PARAM_ORDER if p in avail] or ['RHO']
        cur = self.bg_combo.currentData()
        self.bg_combo.blockSignals(True)
        self.bg_combo.clear()
        for p in params:
            self.bg_combo.addItem(PARAMS[p]['label'], p)
        self.bg_combo.addItem("None", None)
        k = self.bg_combo.findData(cur) if cur else -1
        self.bg_combo.setCurrentIndex(k if k >= 0 else 0)
        self.bg_combo.blockSignals(False)

    def _bg_changed(self, *args):
        if self.line is not None:
            self.load_section(keep_view=True)

    # ------------------------------------------------------------------
    def load_section(self, *args, keep_view=False):
        lyr, f = self._feature()
        if f is None:
            QMessageBox.warning(self, "No profile", "Choose a line layer and a profile (click ↻ to refresh).")
            return
        layers = self.picker.checked_layers()
        if not layers:
            QMessageBox.warning(self, "No sounding layers", "Tick at least one sounding layer.")
            return
        try:
            df, crs = self.registry.combine_soundings(layers)
        except Exception as exc:
            QMessageBox.critical(self, "Sounding data", str(exc))
            return
        g = QgsGeometry(f.geometry())
        if g.isMultipart():
            parts = g.asMultiPolyline()
            pts = max(parts, key=len) if parts else []
        else:
            pts = g.asPolyline()
        if len(pts) < 2:
            QMessageBox.warning(self, "Invalid line", "The profile line needs at least 2 vertices.")
            return
        if lyr.crs().isValid() and crs is not None and crs.isValid() and lyr.crs() != crs:
            tr = QgsCoordinateTransform(lyr.crs(), crs, QgsProject.instance())
            pts = [tr.transform(p) for p in pts]
        line = np.array([[p.x(), p.y()] for p in pts], dtype=float)
        param = self.bg_combo.currentData()
        try:
            sec = compute_section(df, line, self.tol_spin.value(), param or 'RHO')
        except Exception as exc:
            QMessageBox.critical(self, "Section", str(exc))
            return
        if sec is None:
            QMessageBox.information(self, "No soundings",
                                    f"No soundings within {self.tol_spin.value():g} m of the line.")
            return
        new_profile = self.prof_name != self.feat_combo.currentText() or self.line is None
        self.sec, self.line, self.data_crs = sec, line, crs
        self.prof_name = self.feat_combo.currentText()
        if param and (new_profile or param != self._bg_param):
            self._default_bg_range(param)
        self._bg_param = param
        if not keep_view:
            self.view = 'edit'
        self._redraw(reset=not keep_view or new_profile)
        self.status_label.setText(
            f"Section '{self.prof_name}': {sec['n']} soundings along {line_length(line):.0f} m. "
            "Add surfaces and pick them on the section.")

    def _default_bg_range(self, param):
        v = self.sec['vals'][np.isfinite(self.sec['vals'])]
        if param == 'RHO':
            lo, hi = 1.0, 1000.0
        elif v.size:
            lo, hi = np.percentile(v[v > 0] if param != 'C' else v, [2, 98])
        else:
            lo, hi = 0.0, 1.0
        dec = PARAMS[param]['decimals']
        for sp, val in ((self.bg_min, lo), (self.bg_max, hi)):
            sp.blockSignals(True)
            sp.setDecimals(max(dec, 3))
            sp.setValue(float(f"{val:.3g}"))
            sp.blockSignals(False)

    # ------------------------------------------------------------------
    #  Surfaces table
    # ------------------------------------------------------------------
    def _sorted_surfaces(self):
        """Surfaces top → bottom by mean elevation along the section (unpicked last)."""
        if self.sec is None:
            return list(self.surfaces)
        x = np.linspace(0, line_length(self.line), 200)

        def key(s):
            return -float(np.mean(surface_curve(s['pts'], x))) if s['pts'] else math.inf
        return sorted(self.surfaces, key=key)

    def _fill_table(self, select=None):
        self.surfaces = self._sorted_surfaces()
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.surfaces) + 1)
        rows = [(None, "Ground surface", self.top_unit['label'], self.top_unit['color'], "–")]
        rows += [(s, s['name'], s['label'], s['color'], str(len(s['pts']))) for s in self.surfaces]
        for r, (s, name, lab, col, n) in enumerate(rows):
            it = QTableWidgetItem(name)
            if s is None:
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(r, 0, it)
            self.table.setItem(r, 1, QTableWidgetItem(lab))
            ci = QTableWidgetItem("")
            ci.setBackground(QColor(col))
            ci.setFlags(ci.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(r, 2, ci)
            ni = QTableWidgetItem(n)
            ni.setFlags(ni.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(r, 3, ni)
        self.table.blockSignals(False)
        if select is not None and select in self.surfaces:
            self.table.selectRow(self.surfaces.index(select) + 1)

    def _row_unit(self, row):
        """The dict holding row `row` (0 = ground / top formation)."""
        if row <= 0:
            return self.top_unit
        return self.surfaces[row - 1] if row - 1 < len(self.surfaces) else None

    def _current_surface(self):
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        r = rows[0].row() if rows else self.table.currentRow()
        if r is None or r < 1 or r - 1 >= len(self.surfaces):
            return None
        return self.surfaces[r - 1]

    def _table_edited(self, item):
        unit = self._row_unit(item.row())
        if unit is None:
            return
        if item.column() == 0 and unit is not self.top_unit:
            unit['name'] = item.text().strip() or unit['name']
        elif item.column() == 1:
            unit['label'] = item.text().strip() or unit['label']
            if unit.get('auto', True):
                rgb = lith_rgb_or_none(unit['label'])
                if rgb is not None:
                    unit['color'] = '#%02x%02x%02x' % tuple(rgb)
                    self._fill_table(select=unit if unit is not self.top_unit else None)
        self._redraw()

    def _cell_double_clicked(self, row, col):
        if col != 2:
            return
        unit = self._row_unit(row)
        if unit is None:
            return
        c = QColorDialog.getColor(QColor(unit['color']), self, "Formation colour")
        if c.isValid():
            unit['color'] = c.name()
            unit['auto'] = False
            self._fill_table(select=unit if unit is not self.top_unit else None)
            self._redraw()

    def _add_surface(self):
        k = len(self.surfaces) + 1
        used = {self.top_unit['color']} | {s['color'] for s in self.surfaces}
        col = next((c for c in FORMATION_COLORS if c not in used), FORMATION_COLORS[k % len(FORMATION_COLORS)])
        s = dict(name=f"Surface {k}", label=f"Formation {k + 1}", color=col, pts=[])
        self.surfaces.append(s)
        self._fill_table(select=s)
        self.pick_btn.setChecked(True)
        self.status_label.setText(f"'{s['name']}' added – click along the boundary on the section.")
        return s

    def _remove_surface(self):
        s = self._current_surface()
        if s is None:
            QMessageBox.information(self, "Remove", "Select a surface row (not the ground surface).")
            return
        self.surfaces.remove(s)
        self._fill_table()
        self._redraw()

    def _clear_points(self):
        s = self._current_surface()
        if s is None:
            return
        s['pts'] = []
        self._fill_table(select=s)
        self._redraw()

    def _pick_toggled(self, on):
        if on:
            self._set_view('edit')
            if self._toolbar_busy():
                self.status_label.setText("Switch off Zoom / Pan in the plot toolbar to pick points.")
            if self._current_surface() is None:
                if self.surfaces:
                    self.table.selectRow(1)
                else:
                    self._add_surface()
        self.pick_btn.setText("Picking… (click again to stop)" if on else "Pick Points on Section")

    # ------------------------------------------------------------------
    #  Mouse
    # ------------------------------------------------------------------
    def _toolbar_busy(self):
        m = getattr(self.toolbar, 'mode', '')
        return bool(str(getattr(m, 'value', m)))

    def _on_click(self, ev):
        if (self.view != 'edit' or self.sec is None or ev.inaxes is not self.ax or ev.xdata is None
                or not self.pick_btn.isChecked() or self._toolbar_busy()):
            return
        s = self._current_surface()
        if s is None:
            return
        btn = getattr(ev.button, 'value', ev.button)
        Ltot = line_length(self.line)
        if btn == 1:
            d = float(min(max(ev.xdata, 0.0), Ltot))
            s['pts'] = sorted([p for p in s['pts'] if abs(p[0] - d) > 1e-6] + [(d, float(ev.ydata))])
        elif btn == 3 and s['pts']:
            tr = self.ax.transData
            xy = tr.transform(np.asarray(s['pts']))
            dpx = np.hypot(xy[:, 0] - ev.x, xy[:, 1] - ev.y)
            k = int(np.argmin(dpx))
            if dpx[k] <= 20:
                s['pts'].pop(k)
        else:
            return
        self._fill_table(select=s)
        self._redraw()

    def _on_move(self, ev):
        if self.sec is None or ev.inaxes is not self.ax or ev.xdata is None:
            return
        d, z = ev.xdata, ev.ydata
        txt = f"Distance: {d:8.1f} m   Elevation: {z:7.1f} m"
        sec = self.sec
        i = int(np.argmin(np.abs(sec['dist'] - d)))
        if self.view == 'edit' and self.bg_combo.currentData():
            depth = sec['elev'][i] - z
            j = np.where((sec['top'][i] <= depth) & (depth < sec['bot'][i]))[0]
            if len(j) and np.isfinite(sec['vals'][i, j[0]]):
                prm = self.bg_combo.currentData()
                txt += f"   {PARAMS[prm]['label']}: {sec['vals'][i, j[0]]:.4g} {PARAMS[prm]['unit']}"
        else:
            unit = self._unit_at(d, z)
            if unit:
                txt += f"   Formation: {unit}"
        drawn = getattr(self, '_bh_drawn', None)
        if drawn:
            lith = lith_at(drawn[0], drawn[1], d, lambda b: self._bh_y(b, np.array([0.0]))[0] - z)
            if lith:
                txt += f"   BH {lith}"
        self.cursor_label.setText(txt)
        if self.pos_marker is not None and self.iface is not None:
            try:
                x, y = point_at(self.line, d)[0]
                canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
                p = QgsPointXY(float(x), float(y))
                if self.data_crs is not None and self.data_crs.isValid() and self.data_crs != canvas_crs:
                    p = QgsCoordinateTransform(self.data_crs, canvas_crs, QgsProject.instance()).transform(p)
                self.pos_marker.setCenter(p)
                self.pos_marker.show()
            except Exception:
                pass

    def _hide_marker(self):
        if self.pos_marker is not None:
            self.pos_marker.hide()

    def _unit_at(self, d, z):
        x = np.array([d])
        g = np.interp(x, self.sec['dist'], self.sec['elev'])
        if z > g[0]:
            return ""
        label = self.top_unit['label']
        for s, zz in ordered_boundaries(self.surfaces, x, g, np.array([-1e9])):
            if z < zz[0]:
                label = s['label']
        return label

    # ------------------------------------------------------------------
    #  Drawing
    # ------------------------------------------------------------------
    def _set_view(self, v):
        if v == 'geology' and self.pick_btn.isChecked():
            self.pick_btn.setChecked(False)
        self.view = v
        self._redraw()

    def _reset_view(self):
        self._redraw(reset=True)

    def _y_range(self):
        sec = self.sec
        top = float(np.nanmax(sec['elev'])) + MARGIN
        dz = sec['elev'] - sec['doi']
        if np.isfinite(dz).any():
            bottom = float(np.nanmin(dz)) - MARGIN
        else:
            bottom = float(np.nanmin(sec['elev'][:, None] - sec['bot'])) - MARGIN
        for s in self.surfaces:                 # never cut off a picked point
            for _, z in s['pts']:
                bottom = min(bottom, z - 5)
                top = max(top, z + 5)
        return bottom, top

    def _redraw(self, reset=False):
        """Redraw the current view; the zoom is kept unless reset=True."""
        fig = self.figure
        lims = None
        if not reset and self.ax is not None and self.sec is not None:
            lims = (self.ax.get_xlim(), self.ax.get_ylim())
        fig.clf()
        self.ax = None
        if self.sec is None:
            self.canvas.draw_idle()
            return
        has_bg = bool(self.bg_combo.currentData()) and (
            self.view == 'edit' or self.geo_alpha.value() / 100.0 < 0.999)
        self._cax = None
        if has_bg:            # horizontal colorbar in its own row under the section
            from .map_export import panel_gridspec
            gs = panel_gridspec(fig, 1)
            ax = fig.add_subplot(gs[0, :])
            self._cax = fig.add_subplot(gs[1, 1])
            self._cax.set_label('<colorbar>')
        else:
            ax = fig.add_subplot(111)
        self.ax = ax
        sec = self.sec
        Ltot = line_length(self.line)
        y_lo, y_hi = self._y_range()
        bhs = self._boreholes()
        bh_w = self.bh_width_spin.value() or Ltot * 0.012
        # boreholes beyond the line ends sit at their projected distance: widen the section
        x0 = min([0.0] + [b['dist'] - bh_w for b in bhs])
        x1 = max([Ltot] + [b['dist'] + bh_w for b in bhs])
        x = np.linspace(x0, x1, 600)
        ground = np.interp(x, sec['dist'], sec['elev'])
        if self.view == 'edit':
            self._draw_background(ax)
            ax.plot(sec['dist'], sec['elev'], color='black', linewidth=1.0)
            cur = self._current_surface()
            for s, z in ordered_boundaries(self.surfaces, x, ground, np.full(len(x), y_lo - 1e4)):
                sel = s is cur
                ax.plot(x, z, color='black' if sel else '0.25', linewidth=2.2 if sel else 1.2,
                        linestyle='-' if sel else '--', zorder=6)
                p = np.asarray(s['pts'])
                ax.plot(p[:, 0], p[:, 1], 'o', markersize=6 if sel else 4, markerfacecolor=s['color'],
                        markeredgecolor='black', zorder=7)
                k = int(np.argmax(x >= p[:, 0].max())) if p[:, 0].max() < Ltot else len(x) - 1
                ax.text(x[k], z[k], f"  {s['name']}", fontsize=8, va='center', ha='left' if k < len(x) - 1
                        else 'right', zorder=8,
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))
            title = "Edit surfaces"
        else:
            alpha = self.geo_alpha.value() / 100.0
            self._bg_drawn = alpha < 0.999 and bool(self.bg_combo.currentData())
            if self._bg_drawn:
                self._draw_background(ax)
            self._draw_geology(ax, x, ground, y_lo, alpha)
            title = "Geological section"
        if self.snd_check.isChecked():
            ax.plot(sec['dist'], sec['elev'], 'kv', markersize=4 if sec['n'] <= 150 else 2, zorder=9)
        self._draw_bhs(ax, bhs, bh_w)
        ax.set_xlim(x0 - 0.01 * Ltot, x1 + 0.01 * Ltot)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Elevation (m)")
        ax.grid(True, linestyle='--', alpha=0.3)
        st, en = self._direction()
        ax.text(0.0, 1.01, st, transform=ax.transAxes, ha='left', va='bottom', fontsize=12, fontweight='bold')
        ax.text(1.0, 1.01, en, transform=ax.transAxes, ha='right', va='bottom', fontsize=12, fontweight='bold')
        ax.text(0.5, 1.01, f"{title} – {self.prof_name}", transform=ax.transAxes, ha='center',
                va='bottom', fontsize=9, color='0.3')
        fig.tight_layout()
        if lims is not None:
            ax.set_xlim(*lims[0])
            ax.set_ylim(*lims[1])
        self.canvas.draw_idle()

    def _draw_formation_legend(self, ax):
        handles = getattr(self, '_formation_handles', None)
        if not handles:
            return
        if getattr(self, '_bg_drawn', False):   # colour bar on the right: legend inside the plot
            leg = ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.85,
                            title="Formations", title_fontsize=9)
            leg.set_zorder(20)
        else:
            ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8,
                      frameon=False, title="Formations", title_fontsize=9)

    def _boreholes(self):
        if not self.bh_check.isChecked() or self.line is None:
            return []
        try:
            return boreholes_near_line(self.registry, self.bh_picker.checked_layers(), self.line,
                                       self.data_crs, self.bh_dist_spin.value())
        except Exception:
            return []

    def _bh_y(self, b, depths):
        e = b['elev'] if np.isfinite(b['elev']) else float(np.interp(b['dist'], self.sec['dist'],
                                                                      self.sec['elev']))
        return e - depths

    def _draw_bhs(self, ax, bhs, w):
        self._bh_drawn = None
        if not bhs:
            return
        used = draw_boreholes(ax, bhs, w, self._bh_y, labels=True, zorder=10)
        self._bh_drawn = (bhs, w)
        if used:
            from matplotlib.patches import Patch
            handles = [Patch(facecolor=c, edgecolor='black', linewidth=0.4, label=n)
                       for n, c in list(used.items())[:15]]
            from matplotlib.legend import Legend
            leg = Legend(ax, handles, [h.get_label() for h in handles], loc='lower right', fontsize=7,
                         framealpha=0.85, ncol=min(3, (len(handles) + 4) // 5), title="Boreholes",
                         title_fontsize=7)
            leg.set_zorder(20)
            ax.add_artist(leg)          # inside the plot; the formations legend stays the main one

    def _direction(self):
        if self.line is None:
            return "A", "B"
        (x0, y0), (x1, y1) = self.line[0], self.line[-1]
        az = math.degrees(math.atan2(x1 - x0, y1 - y0))
        names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        c = lambda a: names[int(((a % 360) + 22.5) // 45) % 8]
        return c(az + 180), c(az)

    def _draw_background(self, ax):
        prm = self.bg_combo.currentData()
        sec = self.sec
        if not prm:
            return
        vals = sec['vals'].copy()
        lo, hi = self.bg_min.value(), self.bg_max.value()
        log = prm != 'C' and lo > 0
        if hi <= lo:
            hi = lo * 10 if log else lo + 1
        norm = matplotlib.colors.LogNorm(lo, hi) if log else matplotlib.colors.Normalize(lo, hi)
        if log:
            vals = np.where(vals > 0, vals, np.nan)
        cmap = get_colormap(self.bg_cmap.currentText()) or get_colormap('BlueToPink')
        alpha = self.bg_alpha.value() / 100.0
        mask_alpha = self.doi_alpha.value() / 100.0
        d = sec['dist']
        gaps = np.diff(d) if len(d) > 1 else np.array([])
        med = float(np.median(gaps[gaps > 0])) if (gaps > 0).any() else 10.0
        bw = self.bar_spin.value()
        if bw > 0:
            left, right = d - bw / 2.0, d + bw / 2.0
        elif len(d) > 1:
            # each sounding fills half-way to its neighbours (max. 2 × median spacing each side)
            edges = np.concatenate([[d[0] - med / 2], (d[:-1] + d[1:]) / 2, [d[-1] + med / 2]])
            left = np.maximum(edges[:-1], d - 2 * med)
            right = np.minimum(edges[1:], d + 2 * med)
        else:
            left, right = d - 10, d + 10
        n, L = vals.shape
        yt = sec['elev'][:, None] - sec['top']
        yb = sec['elev'][:, None] - sec['bot']
        artist = None
        if self.interp_check.isChecked() and n > 1:
            y0, y1 = float(np.nanmin(yb)), float(np.nanmax(yt))
            ny = int(min(max((y1 - y0) / 0.5, 100), 1500))
            yi = np.linspace(y0, y1, ny + 1)
            yc = (yi[:-1] + yi[1:]) / 2.0
            vv = np.log10(vals) if log else vals
            R = np.full((ny, n), np.nan)
            lo_y, hi_y = np.minimum(yt, yb), np.maximum(yt, yb)
            for j in range(L):
                m = (yc[:, None] > lo_y[None, :, j]) & (yc[:, None] <= hi_y[None, :, j])
                R = np.where(m, vv[None, :, j], R)
            x0, x1 = d[0] - med / 2.0, d[-1] + med / 2.0
            nx = int(min(max((x1 - x0) / max(med / 3.0, 0.5), 200), 3000))
            xi = np.linspace(x0, x1, nx + 1)
            xc = (xi[:-1] + xi[1:]) / 2.0
            k = np.clip(np.searchsorted(d, xc) - 1, 0, n - 2)
            span = d[k + 1] - d[k]
            a = np.clip(np.where(span > 0, (xc - d[k]) / np.where(span > 0, span, 1), 0.0), 0.0, 1.0)
            Z = R[:, k] * (1 - a) + R[:, k + 1] * a
            near = np.minimum(np.abs(xc - d[k]), np.abs(xc - d[k + 1]))
            big_gap = (span > 4 * med) & (near > med)          # do not bridge large gaps
            Z[:, big_gap] = np.nan
            Z[:, (xc < d[0] - med / 2) | (xc > d[-1] + med / 2)] = np.nan
            if log:
                Z = 10 ** Z
            artist = ax.pcolormesh(xi, yi, np.ma.masked_invalid(Z), shading='flat', cmap=cmap, norm=norm,
                                   alpha=alpha, zorder=1)
            if self.frames_check.isChecked():
                # 'auto' width would tile the frames edge to edge: use a slim column instead
                fl, fr = (left, right) if bw > 0 else (d - 0.15 * med, d + 0.15 * med)
                draw_sounding_frames(ax, fl, fr, yt, yb, zorder=4)
        else:
            lx = np.repeat(left[:, None], L, axis=1)
            rx = np.repeat(right[:, None], L, axis=1)
            keep = np.isfinite(yt) & np.isfinite(yb) & np.isfinite(vals)
            if keep.any():
                verts = np.stack([np.stack([lx[keep], yb[keep]], 1), np.stack([rx[keep], yb[keep]], 1),
                                  np.stack([rx[keep], yt[keep]], 1), np.stack([lx[keep], yt[keep]], 1)], axis=1)
                artist = matplotlib.collections.PolyCollection(verts, cmap=cmap, norm=norm, edgecolors='none',
                                                               linewidths=0, alpha=alpha, zorder=1)
                artist.set_array(vals[keep])
                ax.add_collection(artist)
            if self.frames_check.isChecked():
                draw_sounding_frames(ax, left, right, yt, yb, zorder=4)
        cax = getattr(self, '_cax', None)
        if artist is not None:
            if cax is not None:
                self.figure.colorbar(artist, cax=cax, orientation='horizontal', label=PARAMS[prm]['cb'])
            else:
                self.figure.colorbar(artist, ax=ax, orientation='horizontal', label=PARAMS[prm]['cb'],
                                     pad=0.12, fraction=0.05, shrink=0.45)
        elif cax is not None:
            cax.set_visible(False)
        okd = np.isfinite(sec['doi'])
        if okd.any():
            dz = sec['elev'] - sec['doi']
            y_floor = float(np.nanmin(yb)) - 1e3
            if self.interp_check.isChecked() and okd.sum() > 1:
                xs = np.concatenate([[d[okd][0] - med / 2], d[okd], [d[okd][-1] + med / 2]])
                ys = np.concatenate([[dz[okd][0]], dz[okd], [dz[okd][-1]]])
                if self.doi_check.isChecked() and mask_alpha > 0:
                    ax.fill_between(xs, ys, y_floor, color='white', alpha=mask_alpha, linewidth=0, zorder=1.5)
                ax.plot(xs, ys, color='black', linestyle='--', linewidth=0.8, zorder=4)
            else:
                if self.doi_check.isChecked() and mask_alpha > 0:
                    mverts = np.stack([np.stack([left[okd], dz[okd]], 1), np.stack([right[okd], dz[okd]], 1),
                                       np.stack([right[okd], np.full(okd.sum(), y_floor)], 1),
                                       np.stack([left[okd], np.full(okd.sum(), y_floor)], 1)], axis=1)
                    ax.add_collection(matplotlib.collections.PolyCollection(
                        mverts, facecolors='white', edgecolors='none', alpha=mask_alpha, zorder=1.5))
                segs = np.stack([np.stack([left[okd], dz[okd]], 1), np.stack([right[okd], dz[okd]], 1)], axis=1)
                ax.add_collection(matplotlib.collections.LineCollection(
                    segs, colors='black', linestyles='--', linewidths=0.8, zorder=4))

    def _draw_geology(self, ax, x, ground, y_lo, alpha=1.0):
        bottom = np.full(len(x), y_lo - 1e3)
        bounds = ordered_boundaries(self.surfaces, x, ground, bottom)
        units = [(self.top_unit, ground)] + [(s, z) for s, z in bounds]
        handles = []
        for k, (unit, upper) in enumerate(units):
            lower = units[k + 1][1] if k + 1 < len(units) else bottom
            ax.fill_between(x, lower, upper, where=upper > lower, color=unit['color'], linewidth=0,
                            zorder=2, interpolate=True, alpha=alpha)
            handles.append(mpatches.Patch(facecolor=unit['color'], edgecolor='black', linewidth=0.5,
                                          label=unit['label']))
            if self.labels_check.isChecked():
                lo_vis = np.maximum(lower, y_lo)
                thick = upper - lo_vis
                n = len(x)
                inner = np.zeros(n, bool)
                inner[int(n * 0.1):int(n * 0.9)] = True      # keep labels away from the ends
                score = np.where(inner, thick, -np.inf)
                if np.nanmax(thick) > 0:
                    i = int(np.nanargmax(score)) if np.isfinite(score).any() and np.nanmax(score) > 0 \
                        else int(np.nanargmax(thick))
                    ax.text(x[i], (upper[i] + lo_vis[i]) / 2.0, unit['label'], ha='center', va='center',
                            fontsize=9, zorder=8, clip_on=True,
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.6))
        if self.surf_lines_check.isChecked():
            for s, z in bounds:
                ax.plot(x, z, color='black', linewidth=0.9, zorder=5)
        ax.plot(x, ground, color='black', linewidth=1.4, zorder=6)
        ax.fill_between(x, ground, ground + 1e4, color='white', zorder=3)   # sky
        self._formation_handles = handles
        if self.legend_check.isChecked():
            self._draw_formation_legend(ax)

    # ------------------------------------------------------------------
    #  Output
    # ------------------------------------------------------------------
    def _export_surfaces(self):
        picked = [s for s in self.surfaces if s['pts']]
        if self.sec is None or not picked:
            QMessageBox.information(self, "Surfaces", "Load a section and pick at least one surface first.")
            return
        crs = self.data_crs if (self.data_crs is not None and self.data_crs.isValid()) \
            else QgsProject.instance().crs()
        name = f"Surfaces - {self.prof_name}"
        lyr = QgsVectorLayer(f"LineStringZ?crs={crs.authid()}", name, "memory")
        lyr.dataProvider().addAttributes([QgsField("surface", FIELD_STRING), QgsField("below", FIELD_STRING),
                                          QgsField("above", FIELD_STRING), QgsField("profile", FIELD_STRING),
                                          QgsField("z_mean", FIELD_DOUBLE)])
        lyr.updateFields()
        Ltot = line_length(self.line)
        x = np.linspace(0, Ltot, 600)
        ground = np.interp(x, self.sec['dist'], self.sec['elev'])
        bounds = ordered_boundaries(self.surfaces, x, ground, np.full(len(x), -1e9))
        feats, above = [], self.top_unit['label']
        for s, z in bounds:
            p = np.asarray(s['pts'])
            m = (x >= p[:, 0].min()) & (x <= p[:, 0].max())
            if m.sum() < 2:
                m = np.ones(len(x), bool)
            xy = point_at(self.line, x[m])
            wkt = "LineStringZ (" + ", ".join(f"{a:.3f} {b:.3f} {c:.2f}" for (a, b), c in zip(xy, z[m])) + ")"
            f = QgsFeature(lyr.fields())
            f.setGeometry(QgsGeometry.fromWkt(wkt))
            f.setAttributes([s['name'], s['label'], above, self.prof_name, float(np.mean(z[m]))])
            feats.append(f)
            above = s['label']
        lyr.dataProvider().addFeatures(feats)
        lyr.updateExtents()
        QgsProject.instance().addMapLayer(lyr)
        self.status_label.setText(f"{len(feats)} surface(s) added to the map as '{name}' (Z = elevation).")

    def _save_image(self):
        if self.sec is None or self.ax is None:
            QMessageBox.information(self, "Save image", "Load a section first.")
            return
        base = (self.prof_name or "section").replace(" ", "_")
        base += "_geology" if self.view == 'geology' else "_surfaces"
        path, flt = QFileDialog.getSaveFileName(
            self, "Save image", f"{base}.png",
            "PNG (*.png);;PDF (*.pdf);;PDF with location map (*.pdf);;SVG (*.svg)")
        if not path:
            return
        flt = flt or ''
        ext = '.pdf' if 'PDF' in flt else ('.svg' if 'SVG' in flt else '.png')
        if not path.lower().endswith(('.png', '.pdf', '.svg')):
            path += ext
        try:
            if 'map' in flt:
                from .map_export import save_item_pdf
                choices = ["A4 landscape", "A4 portrait"]
                choice, ok = QInputDialog.getItem(self, "Section + location map", "Page format:",
                                                  choices, 0, False)
                if not ok:
                    return
                orient = 'landscape' if choice == choices[0] else 'portrait'
                title = f"{'Geological section' if self.view == 'geology' else 'Section'} – {self.prof_name}"
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                try:
                    save_item_pdf(path, self._build_report_item(title, [orient]), orient)
                finally:
                    QApplication.restoreOverrideCursor()
            else:
                self.figure.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        except Exception as exc:
            QMessageBox.critical(self, "Save image", str(exc))
            return
        self.status_label.setText(f"Image saved to {path}")

    def _build_report_item(self, name, orientations=None):
        """Location map + the section as it is drawn now, for each A4 page format."""
        from .map_export import build_report_item
        drawn = getattr(self, '_bh_drawn', None)
        st, en = self._direction()
        bh_on = self.bh_check.isChecked()
        crs_id = self.data_crs.authid() if (self.data_crs is not None and self.data_crs.isValid()) else ""
        item = build_report_item(
            self.figure, name, self.line, self.data_crs, self.iface, (st, en),
            soundings_xy=self.sec.get('xy'), boreholes=drawn[0] if drawn else None,
            bh_dist=self.bh_dist_spin.value() if bh_on else None, orientations=orientations,
            before=self._hide_marker, kind='geology', length_m=float(line_length(self.line)),
            n_snd=int(self.sec['n']), search_w=float(self.tol_spin.value()), crs=crs_id,
            layers="; ".join(l.name() for l in self.picker.checked_layers()),
            yaxis='Elevation', params='Geological formations')
        self.canvas.draw_idle()
        return item

    def _save_model_layer(self):
        """Store the geological section (and the interpretation) as a feature of the
        'TEM-Viewer geological models' layer."""
        if self.sec is None or self.line is None:
            QMessageBox.information(self, "Save model", "Load a section first.")
            return
        proj = QgsProject.instance()
        crs = self.data_crs if (self.data_crs is not None and self.data_crs.isValid()) else proj.crs()
        from .map_export import add_layer_to_group, unique_layer_name, count_layers_of_kind
        n_existing = count_layers_of_kind(GEO_KIND)
        name, ok = QInputDialog.getText(self, "Save geological model", "Name of the model:",
                                        text=f"Geological model {n_existing + 1} – {self.prof_name}")
        if not ok:
            return
        name = name.strip() or f"Geological model {n_existing + 1}"
        lyr = QgsVectorLayer(f"LineString?crs={crs.authid()}", unique_layer_name(name, GEO_KIND), "memory")
        lyr.dataProvider().addAttributes([
            QgsField("name", FIELD_STRING), QgsField("profile", FIELD_STRING),
            QgsField("length_m", FIELD_DOUBLE), QgsField("search_w", FIELD_DOUBLE),
            QgsField("layers", FIELD_STRING), QgsField("uid", FIELD_STRING),
            QgsField("state", FIELD_STRING), QgsField("report_data", FIELD_STRING)])
        lyr.updateFields()
        lyr.setCrs(crs)
        from .map_export import new_uid
        uid, report_json, warn = new_uid(), "", ""          # no report pages in TEM-Viewer
        f = QgsFeature(lyr.fields())
        f.setGeometry(QgsGeometry.fromPolylineXY([QgsPointXY(float(x), float(y)) for x, y in self.line]))
        vals = {"name": name, "profile": self.prof_name, "length_m": float(line_length(self.line)),
                "search_w": float(self.tol_spin.value()),
                "layers": "; ".join(l.name() for l in self.picker.checked_layers()),
                "uid": uid, "state": json.dumps(self._state()), "report_data": report_json}
        for key, val in vals.items():
            idx = lyr.fields().indexOf(key)
            if idx >= 0:
                f.setAttribute(idx, val)
        lyr.dataProvider().addFeatures([f])
        lyr.updateExtents()
        from .map_export import persist_layer
        try:
            lyr, path = persist_layer(lyr, "TEM-Viewer geological models", GEO_KIND)
            where = f" It is stored in {path}, so it survives reopening the project."
        except Exception as exc:
            lyr.setCustomProperty("temviewer/kind", GEO_KIND)
            add_layer_to_group(lyr, "TEM-Viewer geological models")
            where = (f" WARNING: it could not be written to a file ({exc}) and is only a temporary layer - "
                     "use right-click > Make Permanent before closing the project.")
        lyr.triggerRepaint()
        self.status_label.setText(
            f"Geological model '{name}' saved as its own layer in the group 'TEM-Viewer geological models'"
            "."
            + where + warn)

    def _state(self):
        lyr, f = self._feature()
        return dict(profile_layer=lyr.name() if lyr is not None else "", profile=self.prof_name,
                    search_width=self.tol_spin.value(),
                    line=self.line.tolist() if self.line is not None else None,
                    ground=dict(label=self.top_unit['label'], color=self.top_unit['color'],
                                auto=self.top_unit.get('auto', True)),
                    surfaces=[dict(name=s['name'], label=s['label'], color=s['color'], auto=s.get('auto', True),
                                   points=[list(map(float, p)) for p in s['pts']]) for s in self.surfaces])

    def _save_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save interpretation", f"{self.prof_name or 'geomodel'}.json",
                                              "JSON (*.json)")
        if not path:
            return
        if not path.lower().endswith('.json'):
            path += '.json'
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(self._state(), fh, indent=1)
        self.status_label.setText(f"Interpretation saved to {path}")

    def _load_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load interpretation", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, encoding='utf-8') as fh:
                st = json.load(fh)
            self.apply_state(st)
        except Exception as exc:
            QMessageBox.critical(self, "Load", str(exc))
            return
        self.status_label.setText(f"Interpretation loaded from {path}"
                                  + ("" if self.sec is not None else " – load the section to see it."))

    def apply_state(self, st):
        self.top_unit = dict(label=st['ground']['label'], color=st['ground']['color'],
                             auto=st['ground'].get('auto', True))
        self.surfaces = [dict(name=s['name'], label=s['label'], color=s['color'], auto=s.get('auto', True),
                              pts=[tuple(p) for p in s['points']]) for s in st.get('surfaces', [])]
        k = self.feat_combo.findText(st.get('profile', '')) if st.get('profile') else -1
        if k >= 0:
            self.feat_combo.setCurrentIndex(k)
        self._fill_table()
        self._redraw()

    # ------------------------------------------------------------------
    def showEvent(self, event):
        # pick up profile lines saved meanwhile (e.g. from the Profile tab)
        try:
            cur_feat = self.feat_combo.currentText()
            self._refresh_lines()
            k = self.feat_combo.findText(cur_feat) if cur_feat else -1
            if k >= 0:
                self.feat_combo.setCurrentIndex(k)
        except Exception:
            pass
        super().showEvent(event)

    def stop_tools(self):
        if self.pick_btn.isChecked():
            self.pick_btn.setChecked(False)
        self._hide_marker()

    def set_map_items_visible(self, visible):
        if not visible:
            self._hide_marker()

    def cleanup(self):
        if self.pos_marker is not None and self.iface is not None:
            try:
                self.iface.mapCanvas().scene().removeItem(self.pos_marker)
            except Exception:
                pass
            self.pos_marker = None
