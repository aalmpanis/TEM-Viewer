# -*- coding: utf-8 -*-
"""TEM-Viewer - main window: TEM data, Profile, Mean maps, Geological Model."""
import os
import re
import traceback
import numpy as np
import pandas as pd
import matplotlib.colors
import matplotlib.collections
import matplotlib.patches as mpatches

from qgis.PyQt.QtCore import Qt, QTimer
from qgis.PyQt.QtGui import QColor
from .tv_compat import (QAction, FIELD_DOUBLE, FIELD_INT, FIELD_STRING, qgis_enum, mpl_cmap,
                             FigureCanvasQTAgg as FigureCanvas,
                             NavigationToolbar2QT as NavigationToolbar)
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QMessageBox, QGroupBox,
    QComboBox, QListWidget, QListWidgetItem, QDialogButtonBox,
    QTextEdit, QCheckBox, QTabWidget, QSplitter, QInputDialog, QDoubleSpinBox,
    QWidget
)
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
class SafeFigureCanvas(FigureCanvas):
    """
    Defensive wrapper that ignores non-positional events being
    incorrectly dispatched to enterEvent (QPaintEvent, etc.).
    """
    def enterEvent(self, event):
        if not hasattr(event, 'pos'):
            event.ignore()
            return
        super().enterEvent(event)
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry,
    QgsPointXY, QgsField, QgsPalLayerSettings, QgsVectorLayerSimpleLabeling,
    QgsMarkerSymbol, QgsLineSymbol, QgsSingleSymbolRenderer,
    QgsLayerTreeLayer, QgsProperty, QgsSymbolLayer,
    QgsCategorizedSymbolRenderer, QgsRendererCategory, QgsRendererRange,
    QgsGraduatedSymbolRenderer,
    QgsSpatialIndex, QgsWkbTypes,
    QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsMapLayer, Qgis
)

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
    from .tv_data import canonical_column_names
    df = df.rename(columns=canonical_column_names(list(df.columns)))
    for col in df.columns:
        if col in ('DATE', 'TIME'):
            continue                      # keep as text
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # dummy value (/DUMMY 9999) -> NaN
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace(list(dummy_values), np.nan)
    return df


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


class TemViewerDialog(QDialog):
    ATTR_CLASSES = 15            # colour classes of the sounding attribute map

    def __init__(self, parent=None, iface=None):
        super().__init__(parent)
        self.iface = iface
        self.setWindowTitle("TEM-Viewer")
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowMinimizeButtonHint |
                            Qt.WindowType.WindowMaximizeButtonHint | Qt.WindowType.WindowCloseButtonHint)
        self.resize(1150, 780)
        self.models_df = None
        self._data_crs = None
        self._torn_down = False
        from .tv_data import DataRegistry
        self.registry = DataRegistry(self)
        self._setup_ui()

    def showEvent(self, event):
        self.profile_widget.set_map_items_visible(True)
        self.mr_widget.set_map_items_visible(True)
        super().showEvent(event)

    def _teardown(self):
        if self._torn_down:
            return
        self._torn_down = True
        for name in ('profile_widget', 'mr_widget', 'geo_widget', 'log_link'):
            part = getattr(self, name, None)
            if part is None:
                continue
            for meth in ('stop_picking', 'stop_tools', 'cleanup'):
                if hasattr(part, meth):
                    try:
                        getattr(part, meth)()
                    except Exception:
                        pass
        for fn in (self.remove_map_colorbar, self._attr_timer.stop, self.registry.disconnect_project):
            try:
                fn()
            except Exception:
                pass

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        self.main_tabs = QTabWidget()
        layout.addWidget(self.main_tabs, 1)
        # logs panel: soundings selected / identified on the map
        self.log_link = None
        if self.iface is not None:
            from .log_link import LogLink
            self.log_link = LogLink(self.iface, self.registry, self)

        # =================================================================
        #  TAB 1 - Soundings (import .xyz files as sounding layers)
        # =================================================================
        from .tv_data import LayerPicker
        snd_tab = QWidget()
        st_v = QVBoxLayout(snd_tab)
        imp_grp = QGroupBox("Import Soundings")
        ig = QGridLayout(imp_grp)
        self.xyz_edit = QLineEdit()
        self.xyz_edit.setPlaceholderText(".xyz model file(s) – with a commented header or plain column files")
        self.xyz_btn = QPushButton("Browse…")
        self.xyz_btn.clicked.connect(self._pick_xyz)
        self.load_btn = QPushButton("Load as Layer(s)")
        self.load_btn.clicked.connect(self._load_xyz_layers)
        _pcrs = QgsProject.instance().crs()
        self.xyz_crs_edit = QLineEdit(_pcrs.authid() if _pcrs.isValid() else "")
        self.xyz_crs_edit.setPlaceholderText("e.g. EPSG:32632")
        self.xyz_crs_edit.setMaximumWidth(120)
        self.xyz_crs_edit.setToolTip("Used only when the file does not declare its coordinate system "
                                     "(default: the project CRS)")
        ig.addWidget(QLabel("Files (.xyz):"), 0, 0)
        ig.addWidget(self.xyz_edit, 0, 1)
        ig.addWidget(self.xyz_btn, 0, 2)
        ig.addWidget(QLabel("CRS if not in file:"), 1, 0)
        ig.addWidget(self.xyz_crs_edit, 1, 1)
        ig.addWidget(self.load_btn, 1, 2)
        st_v.addWidget(imp_grp)

        lst_grp = QGroupBox("Sounding Layers in the Project")
        lg = QVBoxLayout(lst_grp)
        self.snd_layers_table = QTableWidget(0, 5)
        self.snd_layers_table.setHorizontalHeaderLabels(["Layer", "Soundings", "Layers/model", "CRS", "Source"])
        self.snd_layers_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.snd_layers_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.snd_layers_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.snd_layers_table.horizontalHeader().setStretchLastSection(True)
        lg.addWidget(self.snd_layers_table)
        rm_h = QHBoxLayout()
        rm_h.addWidget(QLabel("Select one or more rows (Ctrl/Shift+click) to remove those layers."))
        rm_h.addStretch()
        self.remove_snd_btn = QPushButton("Remove Selected Layer(s)")
        self.remove_snd_btn.clicked.connect(self._remove_sounding_layers)
        rm_h.addWidget(self.remove_snd_btn)
        lg.addLayout(rm_h)
        st_v.addWidget(lst_grp, 1)

        attr_grp = QGroupBox("Sounding Attribute Map")
        a_grid = QGridLayout()
        self.attr_layer_combo = QComboBox()
        self.attr_layer_combo.currentIndexChanged.connect(self._on_attr_layer_changed)
        self.attr_combo = QComboBox()
        self.attr_map_btn = QPushButton("Plot Soundings on Map")
        self.attr_map_btn.clicked.connect(self._run_attribute_map)
        self.attr_combo.currentIndexChanged.connect(self._attr_auto_range)
        from .resistivity_tool import get_colormap as _gc
        self.attr_cmap_combo = QComboBox()
        self.attr_cmap_combo.setEditable(True)
        self.attr_cmap_combo.addItems([c for c in ('BlueToPink', 'Spectral', 'viridis', 'plasma', 'turbo',
                                                   'RdYlGn', 'RdYlBu', 'coolwarm', 'jet', 'cividis', 'gray')
                                       if _gc(c) is not None])
        self.attr_cmap_combo.setCurrentText('viridis')
        self.attr_rev_check = QCheckBox("Reverse")
        self.attr_scale_combo = QComboBox()
        self.attr_scale_combo.addItems(["Linear", "Logarithmic"])
        self.attr_min_spin, self.attr_max_spin = QDoubleSpinBox(), QDoubleSpinBox()
        for sp in (self.attr_min_spin, self.attr_max_spin):
            sp.setRange(-1e9, 1e9)
            sp.setDecimals(3)
            sp.setKeyboardTracking(False)
        self.attr_auto_btn = QPushButton("Auto")
        self.attr_auto_btn.setToolTip("Min/max from the data (2–98 %)")
        self.attr_auto_btn.clicked.connect(lambda: self._attr_auto_range(force=True))
        self.attr_symbol_combo = QComboBox()
        self.attr_symbol_combo.addItems(["circle", "square", "triangle", "diamond", "star", "cross"])
        self.attr_size_spin = QDoubleSpinBox()
        self.attr_size_spin.setRange(0.2, 20)
        self.attr_size_spin.setSingleStep(0.5)
        self.attr_size_spin.setValue(2.5)
        self.attr_size_spin.setSuffix(" mm")
        self.attr_size_spin.setKeyboardTracking(False)
        a_grid.addWidget(QLabel("Layer:"), 0, 0)
        a_grid.addWidget(self.attr_layer_combo, 0, 1, 1, 3)
        a_grid.addWidget(QLabel("Color by:"), 0, 4)
        a_grid.addWidget(self.attr_combo, 0, 5, 1, 2)
        a_grid.addWidget(QLabel("Colormap:"), 1, 0)
        a_grid.addWidget(self.attr_cmap_combo, 1, 1)
        a_grid.addWidget(self.attr_rev_check, 1, 2)
        a_grid.addWidget(QLabel("Scale:"), 1, 3)
        a_grid.addWidget(self.attr_scale_combo, 1, 4)
        a_grid.addWidget(QLabel("Min / Max:"), 2, 0)
        a_grid.addWidget(self.attr_min_spin, 2, 1)
        a_grid.addWidget(self.attr_max_spin, 2, 2)
        a_grid.addWidget(self.attr_auto_btn, 2, 3)
        a_grid.addWidget(QLabel("Symbol:"), 3, 0)
        a_grid.addWidget(self.attr_symbol_combo, 3, 1)
        a_grid.addWidget(QLabel("Size:"), 3, 2)
        a_grid.addWidget(self.attr_size_spin, 3, 3)
        # colour bar: preview in the tab + optional legend on the map canvas
        from .colorbar_widget import ColorbarWidget, MapColorbar
        self.attr_map_cbar_check = QCheckBox("Colour bar on map")
        self.attr_map_cbar_check.setChecked(True)
        self.attr_map_cbar_check.setToolTip("Show this colour bar in a corner of the QGIS map; it follows "
                                            "every change and hides when the layer is switched off")
        self.attr_cbar_corner = QComboBox()
        self.attr_cbar_corner.addItems(list(MapColorbar.CORNERS))
        a_grid.addWidget(self.attr_map_cbar_check, 3, 4)
        a_grid.addWidget(self.attr_cbar_corner, 3, 5, 1, 2)
        self.attr_cbar = ColorbarWidget(self, horizontal=True)
        a_grid.addWidget(self.attr_cbar, 4, 0, 1, 7)
        a_grid.addWidget(self.attr_map_btn, 5, 5, 1, 2)
        self._map_cbar = None
        a_grid.setColumnStretch(1, 1)
        a_grid.setColumnStretch(5, 1)
        self._attr_layer_id = None      # map layer made by "Plot Soundings on Map"
        self._attr_timer = QTimer(self)
        self._attr_timer.setSingleShot(True)
        self._attr_timer.setInterval(200)
        self._attr_timer.timeout.connect(self._restyle_attr_layer)
        for sig in (self.attr_cmap_combo.currentTextChanged, self.attr_rev_check.toggled,
                    self.attr_scale_combo.currentIndexChanged, self.attr_min_spin.valueChanged,
                    self.attr_max_spin.valueChanged, self.attr_symbol_combo.currentIndexChanged,
                    self.attr_size_spin.valueChanged):
            sig.connect(lambda *_: self._attr_timer.start())
        for sig in (self.attr_cmap_combo.currentTextChanged, self.attr_rev_check.toggled,
                    self.attr_scale_combo.currentIndexChanged, self.attr_min_spin.valueChanged,
                    self.attr_max_spin.valueChanged, self.attr_combo.currentIndexChanged):
            sig.connect(lambda *_: self._update_attr_colorbar())
        self.attr_map_cbar_check.toggled.connect(
            lambda on: self._map_cbar.set_enabled(on) if self._map_cbar is not None else None)
        self.attr_cbar_corner.currentIndexChanged.connect(
            lambda i: self._map_cbar.set_corner(i) if self._map_cbar is not None else None)
        attr_grp.setLayout(a_grid)
        st_v.addWidget(attr_grp)
        self.main_tabs.addTab(snd_tab, "TEM data")
        self.registry.changed.connect(self._refresh_sounding_overview)

        # =================================================================
        #  Profile - sections along a line / through selected soundings
        # =================================================================
        from .resistivity_tool import ResistivityProfileWidget
        self.profile_widget = ResistivityProfileWidget(self, iface=self.iface, embedded=True,
                                                       registry=self.registry)
        self.main_tabs.addTab(self.profile_widget, "Profile")

        # =================================================================
        #  Mean maps - mean resistivity depth / elevation slices
        # =================================================================
        from .mr_maps import MRMapsWidget
        self.mr_widget = MRMapsWidget(self, iface=self.iface, registry=self.registry)
        self.main_tabs.addTab(self.mr_widget, "Mean maps")

        # =================================================================
        #  Geological Model - surfaces and formations along a profile line
        # =================================================================
        from .geo_model import GeoModelWidget
        self.geo_widget = GeoModelWidget(self, iface=self.iface, registry=self.registry)
        self.main_tabs.addTab(self.geo_widget, "Geological Model")

        self.status_label = QLabel("Load .xyz model files on the TEM data tab, then plot profiles, "
                                   "mean maps or a geological model.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self._refresh_sounding_overview()

    def _pick_xyz(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select .xyz file(s)", "",
                                                "XYZ files (*.xyz);;All files (*)")
        if paths:
            self.xyz_edit.setText("; ".join(paths))

    def _load_xyz_layers(self):
        paths = [p.strip() for p in self.xyz_edit.text().split(";") if p.strip()]
        if not paths:
            QMessageBox.warning(self, "No file", "Choose one or more .xyz files.")
            return
        fallback = QgsCoordinateReferenceSystem(self.xyz_crs_edit.text().strip())
        if not fallback.isValid():
            fallback = QgsProject.instance().crs()
        done = []
        for path in paths:
            if not os.path.isfile(path):
                QMessageBox.warning(self, "Missing file", f"File not found:\n{path}")
                continue
            try:
                df = read_xyz_file(path)
                crs = read_xyz_crs(path) or fallback
                name = f"Soundings - {os.path.splitext(os.path.basename(path))[0]}"
                lyr = self.registry.add_soundings(df, name, crs, path)
                done.append(f"{lyr.name()} ({lyr.featureCount()} soundings, {crs.authid()})")
            except Exception as exc:
                QMessageBox.critical(self, "Error loading .xyz", f"{os.path.basename(path)}:\n{exc}")
        if done:
            self.status_label.setText("Loaded " + "; ".join(done))

    def _refresh_sounding_overview(self, *args):
        layers = self.registry.sounding_layers()
        self.snd_layers_table.setRowCount(len(layers))
        cur = self.attr_layer_combo.currentData()
        self.attr_layer_combo.blockSignals(True)
        self.attr_layer_combo.clear()
        for i, lyr in enumerate(layers):
            n_rho = len([f for f in lyr.fields().names() if re.match(r'RHO_\d+$', f)])
            src = self.registry.soundings.get(lyr.id(), {}).get('path') or lyr.source()
            for j, v in enumerate((lyr.name(), str(lyr.featureCount()), str(n_rho),
                                   lyr.crs().authid(), src)):
                it = QTableWidgetItem(v)
                it.setData(Qt.ItemDataRole.UserRole, lyr.id())
                self.snd_layers_table.setItem(i, j, it)
            self.attr_layer_combo.addItem(lyr.name(), lyr.id())
        if cur:
            k = self.attr_layer_combo.findData(cur)
            if k >= 0:
                self.attr_layer_combo.setCurrentIndex(k)
        self.attr_layer_combo.blockSignals(False)
        self.snd_layers_table.resizeColumnsToContents()
        self._on_attr_layer_changed()

    def _remove_sounding_layers(self):
        rows = sorted({i.row() for i in self.snd_layers_table.selectedItems()})
        ids = [self.snd_layers_table.item(r, 0).data(Qt.ItemDataRole.UserRole) for r in rows]
        layers = [l for l in (QgsProject.instance().mapLayer(i) for i in ids) if l is not None]
        if not layers:
            QMessageBox.information(self, "Remove layers", "Select one or more sounding layers in the table.")
            return
        names = "\n".join(f"  • {l.name()}" for l in layers)
        ans = QMessageBox.question(
            self, "Remove sounding layers",
            f"Remove {len(layers)} layer(s) from the project?\n\n{names}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if ans != QMessageBox.StandardButton.Yes:
            return
        QgsProject.instance().removeMapLayers([l.id() for l in layers])
        self.status_label.setText(f"Removed {len(layers)} sounding layer(s).")

    def _on_attr_layer_changed(self, *args):
        lyr = QgsProject.instance().mapLayer(self.attr_layer_combo.currentData() or '')
        cur = self.attr_combo.currentText()
        self.attr_combo.clear()
        if lyr is None:
            return
        preferred = ['ELEVATION', 'DOI_STANDARD', 'DOI_CONSERVATIVE', 'RESDATA', 'RESTOTAL', 'NUMDATA']
        names = lyr.fields().names()
        up = {n.upper(): n for n in names}
        items = [up[p] for p in preferred if p in up]
        items += sorted((n for n in names if re.match(r'RHO_\d+$', n, re.I)),
                        key=lambda n: int(re.search(r'\d+', n).group()))
        # any other numeric column of the layer (e.g. DOI_*, extra attributes)
        skip = {'X', 'Y', 'RECORD', 'FID', '_FID'}
        numeric = {f.name() for f in lyr.fields() if f.isNumeric()}
        items += sorted(n for n in names if n in numeric and n not in items
                        and n.upper() not in skip and not re.match(r'(DEP_TOP|DEP_BOT|THK)_\d+$', n, re.I))
        self.attr_combo.addItems(items)
        if cur in items:
            self.attr_combo.setCurrentText(cur)

    def _attr_values(self):
        lyr = QgsProject.instance().mapLayer(self.attr_layer_combo.currentData() or '')
        attr = self.attr_combo.currentText()
        if lyr is None or not attr:
            return None
        df = self.registry.soundings_df(lyr)
        if df is None or df.empty or attr not in df.columns:
            return None
        v = pd.to_numeric(df[attr], errors='coerce').to_numpy(dtype=float)
        return v[np.isfinite(v)]

    def _attr_auto_range(self, *args, force=False):
        v = self._attr_values()
        if v is None or not v.size:
            return
        lo, hi = np.percentile(v, [2, 98])
        if hi <= lo:
            lo, hi = float(v.min()), float(v.max())
        if hi <= lo:
            hi = lo + 1.0
        for sp, val in ((self.attr_min_spin, lo), (self.attr_max_spin, hi)):
            sp.blockSignals(True)
            sp.setValue(float(f"{val:.3g}"))
            sp.blockSignals(False)
        if force:
            self._attr_timer.start()
        self._update_attr_colorbar()

    def _attr_cbar_spec(self, title):
        from .resistivity_tool import get_colormap
        from .colorbar_widget import colorbar_spec
        cmap = get_colormap(self.attr_cmap_combo.currentText(), self.attr_rev_check.isChecked()) \
            or get_colormap('viridis')
        lo, hi = self.attr_min_spin.value(), self.attr_max_spin.value()
        if hi <= lo:
            hi = lo + 1.0
        log = self.attr_scale_combo.currentIndex() == 1 and lo > 0
        return colorbar_spec(cmap, lo, hi, log, self.ATTR_CLASSES, title)

    def _update_attr_colorbar(self):
        """Refresh the colour bar in the tab and, for the plotted layer, on the map."""
        try:
            self.attr_cbar.set_spec(self._attr_cbar_spec(self.attr_combo.currentText() or ""))
        except Exception:
            pass
        lyr = QgsProject.instance().mapLayer(self._attr_layer_id or '')
        if lyr is None or self.iface is None:
            return
        try:
            if self._map_cbar is None:
                from .colorbar_widget import MapColorbar
                self._map_cbar = MapColorbar(self.iface.mapCanvas())
                self._map_cbar.set_corner(self.attr_cbar_corner.currentIndex())
                self._map_cbar.set_enabled(self.attr_map_cbar_check.isChecked())
            self._map_cbar.set(lyr.id(), self._attr_cbar_spec(lyr.customProperty("temviewer/attr") or ""))
        except Exception:
            pass

    def _run_attribute_map(self):
        lyr = QgsProject.instance().mapLayer(self.attr_layer_combo.currentData() or '')
        if lyr is None:
            QMessageBox.warning(self, "No layer", "Import or choose a sounding layer first.")
            return
        saved = (self.models_df, self._data_crs)
        try:
            self.models_df = self.registry.soundings_df(lyr)
            self._data_crs = lyr.crs()
            self._plot_soundings_attribute_map()
        finally:
            self.models_df, self._data_crs = saved

    def _plot_soundings_attribute_map(self):
        if self.models_df is None or self.models_df.empty:
            QMessageBox.warning(self, "No data", "Load the .xyz file first.")
            return
        attr = self.attr_combo.currentText()
        if not attr:
            QMessageBox.warning(self, "No attribute", "Select an attribute to color by.")
            return
        if attr not in self.models_df.columns:
            QMessageBox.warning(self, "Missing column", f"Column '{attr}' not found in the .xyz data.")
            return
        x_col = 'X'
        y_col = 'Y'
        df = self.models_df[[x_col, y_col, attr]].copy()
        df[attr] = pd.to_numeric(df[attr], errors='coerce')
        df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
        df = df.dropna()
        if df.empty:
            QMessageBox.warning(self, "No valid data", "No valid coordinates or values to plot.")
            return
        min_v = float(df[attr].min())
        max_v = float(df[attr].max())
        proj = QgsProject.instance()
        crs = self._data_crs if (self._data_crs is not None and self._data_crs.isValid()) else proj.crs()
        crs_str = crs.authid() if crs.isValid() else 'EPSG:4326'
        layer_name = f"Soundings ({attr})"
        for old_lyr in proj.mapLayersByName(layer_name):
            proj.removeMapLayer(old_lyr.id())
        lyr = QgsVectorLayer(f"Point?crs={crs_str}", layer_name, "memory")
        pr = lyr.dataProvider()
        pr.addAttributes([
            QgsField("ID", FIELD_INT),
            QgsField("X", FIELD_DOUBLE),
            QgsField("Y", FIELD_DOUBLE),
            QgsField(attr, FIELD_DOUBLE),
        ])
        lyr.updateFields()
        for i, (_, row) in enumerate(df.iterrows()):
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromPointXY(
                QgsPointXY(float(row[x_col]), float(row[y_col]))
            ))
            f.setAttributes([
                i,
                float(row[x_col]),
                float(row[y_col]),
                float(row[attr])
            ])
            pr.addFeature(f)
        lyr.setCustomProperty("temviewer/attr", attr)
        self._style_attr_layer(lyr, attr, df[attr].to_numpy(dtype=float))
        lyr.triggerRepaint()
        proj.addMapLayer(lyr)
        self._attr_layer_id = lyr.id()
        self._update_attr_colorbar()
        ext = lyr.extent()
        if self.iface is not None and not ext.isEmpty():
            ext.scale(1.2)
            self.iface.mapCanvas().setExtent(ext)
            self.iface.mapCanvas().refresh()
        self.status_label.setText(
            f"Plotted {len(df)} soundings by {attr} (data {min_v:.3g} – {max_v:.3g}). "
            "Colormap, range, symbol and size update the layer directly."
        )

    def _style_attr_layer(self, lyr, attr, values, n_classes=15):
        """Graduated colours from a matplotlib colormap (incl. BlueToPink), linear or log
        classes, chosen symbol and size."""
        from .resistivity_tool import get_colormap
        cmap = get_colormap(self.attr_cmap_combo.currentText(), self.attr_rev_check.isChecked()) \
            or get_colormap('viridis')
        lo, hi = self.attr_min_spin.value(), self.attr_max_spin.value()
        if hi <= lo:
            hi = lo + 1.0
        log = self.attr_scale_combo.currentIndex() == 1 and lo > 0
        edges = (np.logspace(np.log10(lo), np.log10(hi), n_classes + 1) if log
                 else np.linspace(lo, hi, n_classes + 1))
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        if v.size:                                  # open-ended first and last class
            edges[0] = min(edges[0], float(v.min()))
            edges[-1] = max(edges[-1], float(v.max()))
        norm = (matplotlib.colors.LogNorm(lo, hi) if log else matplotlib.colors.Normalize(lo, hi))
        shape = self.attr_symbol_combo.currentText()
        size = self.attr_size_spin.value()
        ranges = []
        for k in range(n_classes):
            a, b = float(edges[k]), float(edges[k + 1])
            mid = np.sqrt(max(a, lo) * min(b, hi)) if log else (max(a, lo) + min(b, hi)) / 2.0
            r, g, bl, _ = cmap(norm(mid))
            props = {'name': shape, 'size': f'{size:.2f}',
                     'color': f'{int(r * 255)},{int(g * 255)},{int(bl * 255)}',
                     'outline_color': '50,50,50', 'outline_width': '0.1'}
            if shape == 'cross':
                props.update({'outline_color': props['color'], 'outline_width': '0.5'})
            sym = QgsMarkerSymbol.createSimple(props)
            ranges.append(QgsRendererRange(a, b, sym, f"{a:.3g} – {b:.3g}"))
        lyr.setRenderer(QgsGraduatedSymbolRenderer(attr, ranges))
        lyr.triggerRepaint()

    def _restyle_attr_layer(self):
        lyr = QgsProject.instance().mapLayer(self._attr_layer_id or '')
        if lyr is None:
            return
        attr = lyr.customProperty("temviewer/attr")
        idx = lyr.fields().indexOf(attr) if attr else -1
        if idx < 0:
            return
        vals = [f.attributes()[idx] for f in lyr.getFeatures()]
        vals = np.array([v if v is not None else np.nan for v in vals], dtype=float)
        self._style_attr_layer(lyr, attr, vals)
        self._update_attr_colorbar()
        if self.iface is not None:
            try:
                self.iface.layerTreeView().refreshLayerSymbology(lyr.id())
            except Exception:
                pass

    def remove_map_colorbar(self):
        if self._map_cbar is not None:
            self._map_cbar.remove()
            self._map_cbar = None

    def closeEvent(self, event):
        self._teardown()
        super().closeEvent(event)

    def done(self, result):
        # closing TEM-Viewer (window X, Esc, Close) ends the session: everything it put
        # on the map apart from the data layers is removed; reopening starts fresh
        self._teardown()
        super().done(result)


class TemViewerPlugin:
    """TEM-Viewer (menu: Plugins > TEM-Viewer, and a toolbar button)."""

    def __init__(self, iface):
        self.iface = iface
        self.dialog = None
        self.action = None

    def initGui(self):
        from qgis.PyQt.QtGui import QIcon
        from .resistivity_tool import register_bluepink_ramp
        register_bluepink_ramp()
        icon = QIcon(os.path.join(os.path.dirname(__file__), 'icon.png'))
        self.action = QAction(icon, "TEM-Viewer", self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addPluginToMenu("TEM-Viewer", self.action)
        self.iface.addToolBarIcon(self.action)

    def unload(self):
        self.iface.removePluginMenu("TEM-Viewer", self.action)
        self.iface.removeToolBarIcon(self.action)
        dlg, self.dialog = self.dialog, None
        if dlg is not None:
            try:
                dlg._teardown()
                dlg.close()
            except Exception:
                pass
            dlg.deleteLater()

    def _dialog_finished(self, *args):
        dlg, self.dialog = self.dialog, None
        if dlg is not None:
            dlg.deleteLater()

    def run(self):
        # one window per session; closing it ends the session and the next start is fresh
        # (the data layers on the map stay)
        if self.dialog is None:
            import time as _t
            t0 = _t.perf_counter()
            self.dialog = TemViewerDialog(self.iface.mainWindow(), iface=self.iface)
            from .tv_data import log_timing
            log_timing("opening the TEM-Viewer window", _t.perf_counter() - t0)
            self.dialog.finished.connect(self._dialog_finished)
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()
