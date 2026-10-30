# -*- coding: utf-8 -*-
"""
TEM-Viewer - shared data layer.

Every dataset lives as a point layer on the map. The registry links each layer to
its full data (all sounding model columns / borehole table + lithology logs), so
any tab can work on any layer(s) the user picks instead of on "the last file
loaded".  Layers TEM-Viewer did not create (e.g. a saved GeoPackage) also work as
long as they carry the needed attribute fields.

Standard column names used everywhere downstream
  soundings : X, Y, RECORD, ELEVATION, RHO_i, DEP_TOP_i, DEP_BOT_i, THK_i, DOI_STANDARD ...
  boreholes : BoreholeNo, XUTM, YUTM, DrillDepth, Elevation (+ any other columns)
  logs      : BoreholeNo, Top, Bottom, LayerDescription, RockType (, Description)
"""
import re
import json

import numpy as np
import pandas as pd

import os
from qgis.PyQt.QtCore import Qt, QObject, pyqtSignal
from .tv_compat import FIELD_DOUBLE, FIELD_STRING
from qgis.PyQt.QtWidgets import QListWidget, QListWidgetItem, QAbstractItemView
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry, QgsPointXY, QgsField,
    QgsWkbTypes, QgsCoordinateTransform, QgsCoordinateReferenceSystem,
    QgsMarkerSymbol, QgsSingleSymbolRenderer, QgsPalLayerSettings,
    QgsVectorLayerSimpleLabeling, Qgis
)

KIND_PROP = "temviewer/kind"          # custom layer property: 'soundings' / 'boreholes'
LOGS_PROP = "temviewer/logs"          # lithology logs of a borehole layer (JSON records)
SOURCE_PROP = "temviewer/source"      # file the boreholes were imported from (.bor / .xlsx)
LOG_FIELD = "lith_log"              # attribute with each borehole's own log (JSON), so the logs
                                    # travel with the features: Save Features As, Make Permanent,
                                    # copies, other projects ...

SND_COLS = {
    'X': ['X', 'XUTM', 'UTMX', 'UTME', 'EASTING', 'EAST', 'E'],
    'Y': ['Y', 'YUTM', 'UTMY', 'UTMN', 'NORTHING', 'NORTH', 'N'],
    'RECORD': ['RECORD', 'SOUNDING_ID', 'SOUNDINGID', 'ID', 'POINTID', 'FID'],
    'ELEVATION': ['ELEVATION', 'ELEV', 'TOPO', 'DEM', 'Z'],
}
BH_FIELDS = {
    'BoreholeNo': ['boreholeno', 'name', 'borehole', 'borehole id', 'hole id', 'id'],
    'DrillDepth': ['drilldepth', 'depth', 'depthm', 'depth_m', 'total depth'],
    'Elevation': ['elevation', 'elev', 'z', 'ground elevation'],
}
LOG_COLS = ['BoreholeNo', 'Top', 'Bottom', 'LayerDescription', 'RockType', 'Description']


def _find(cols, candidates):
    up = {str(c).upper(): c for c in cols}
    for cand in candidates:
        if cand.upper() in up:
            return up[cand.upper()]
    return None


_CANON_RE = re.compile(r'^(RHO|DEP_TOP|DEP_BOT|THK|RHO_STD|DOI_STANDARD|DOI_CONSERVATIVE|'
                       r'DOI_CONSERV|DOI_STD|RESDATA|RESTOTAL|NUMDATA|ELEVATION)(_\d+)?$', re.I)


# IP model columns under other names -> the names used everywhere
#   PHI  maximum phase angle (MPA model)   TPHI  its time constant   C  frequency exponent
#   M0   chargeability (Cole-Cole)         TAU   Cole-Cole time constant
IP_ALIASES = {
    'PHI': ('PHI', 'PHIMAX', 'PHI_MAX', 'MPA', 'PHIM', 'PHI_M'),
    'TPHI': ('TPHI', 'TAUPHI', 'TAU_PHI', 'TAUP', 'T_PHI'),
    'C': ('C', 'CEXP', 'C_EXP', 'C_EXPONENT'),
    'M0': ('M0', 'M', 'CHG', 'CHARG', 'CHARGEABILITY', 'M_0'),
    'TAU': ('TAU', 'TAU_CC', 'TAUCC'),
}


def _ip_alias_names(columns):
    """Rename dict for numbered model columns: upper case, and IP columns under other
    names to the standard ones (phimax_3 -> PHI_3, Chargeability_3 -> M0_3)."""
    alias = {a: std for std, names in IP_ALIASES.items() for a in names}
    groups = {}
    for c in columns:
        m = re.match(r'^([A-Za-z][A-Za-z0-9_]*?)_(\d+)$', str(c))
        if m:
            groups.setdefault(m.group(1).upper(), []).append((c, int(m.group(2))))
    taken = {str(c) for c in columns}
    ren = {}
    for prefix, items in groups.items():
        target = alias.get(prefix, prefix)
        if target != prefix and target in groups:
            target = prefix                       # the standard name is in the file already
        for c, n in items:
            new = f"{target}_{n}"
            if new != str(c) and new not in taken:
                ren[c] = new
                taken.add(new)
    return ren


def canonical_column_names(columns):
    """Upper-case the AEM model columns (RHO_#, DEP_TOP_#, THK_#, DOI_Standard,
    DOI_Conservative, ...) so 'DOI_Standard' and 'DOI_STANDARD' are the same column.
    Returns a rename dict; other columns are left untouched."""
    ren = _ip_alias_names(columns)
    taken = (set(columns) - set(ren)) | set(ren.values())
    columns = [c for c in columns if c not in ren]
    for c in columns:
        if _CANON_RE.match(str(c)) and str(c) != str(c).upper() and str(c).upper() not in taken:
            ren[c] = str(c).upper()
            taken.add(str(c).upper())
    return ren


# Shapefile (.dbf) field names are cut to 10 characters, and duplicates get a
# number: DOI_STANDARD -> DOI_STANDA, DOI_CONSERVATIVE -> DOI_CONSER,
# DEP_BOT_STD_11 -> DEP_BOT_31, ...  Map them back to the standard names.
_TRUNCATED_ALIASES = {
    'DOI_STANDA': 'DOI_STANDARD', 'DOI_STAND': 'DOI_STANDARD', 'DOI_STD': 'DOI_STANDARD',
    'DOI_CONSER': 'DOI_CONSERVATIVE', 'DOI_CONSERV': 'DOI_CONSERVATIVE',
    'DOI_CONS': 'DOI_CONSERVATIVE',
}


def repair_truncated_columns(df):
    """Rename shapefile-truncated AEM columns back to their full names, and move
    numbered model columns beyond the number of layers (DEP_BOT_31.. made by the
    truncation of DEP_BOT_STD_#) out of the way so they are not read as layers."""
    up = {str(c).upper(): c for c in df.columns}
    ren = {}
    for short, full in _TRUNCATED_ALIASES.items():
        if full not in up and short in up and full not in ren.values():
            ren[up[short]] = full
    n_rho = len([c for c in df.columns if re.match(r'RHO_\d+$', str(c), re.I)])
    if n_rho:
        for c in df.columns:
            m = re.match(r'(DEP_TOP|DEP_BOT|THK)_(\d+)$', str(c), re.I)
            if m and int(m.group(2)) > n_rho:
                ren[c] = f"_SHP_{c}"
    return df.rename(columns=ren) if ren else df


def standardize_soundings(df):
    """Add the standard X, Y, RECORD, ELEVATION columns (copies of the detected
    ones) so the rest of the code does not depend on the file's naming."""
    df = df.copy()
    df = df.rename(columns=canonical_column_names(list(df.columns)))
    df = repair_truncated_columns(df)
    if not any(re.match(r'RHO_\d+$', str(c)) for c in df.columns):
        raise ValueError("No resistivity columns (RHO_1, RHO_2, ...) found.")
    for std, cands in SND_COLS.items():
        src = _find(df.columns, cands)
        if src is None:
            if std == 'RECORD':
                df['RECORD'] = np.arange(1, len(df) + 1)
                continue
            if std == 'ELEVATION':
                df['ELEVATION'] = 0.0
                continue
            raise ValueError(f"No {std} coordinate column found (tried {', '.join(cands)}).")
        if src != std:
            df[std] = df[src]
        df[std] = pd.to_numeric(df[std], errors='coerce')
    df = df.dropna(subset=['X', 'Y', 'RECORD'])
    df['RECORD'] = df['RECORD'].astype(int)
    return df.reset_index(drop=True)


def logs_to_field_values(logs, names):
    """{borehole name: compact JSON of its log} for the LOG_FIELD attribute."""
    out = {}
    if logs is None or logs.empty:
        return out
    want = set(str(n) for n in names)
    lg = logs[logs['BoreholeNo'].astype(str).isin(want)]

    def _v(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return float(x) if isinstance(x, (int, float, np.integer, np.floating)) else str(x)
    for name, sub in lg.groupby(lg['BoreholeNo'].astype(str), sort=False):
        rows = []
        for _, r in sub.iterrows():
            rows.append([_v(pd.to_numeric(r.get('Top'), errors='coerce')),
                         _v(pd.to_numeric(r.get('Bottom'), errors='coerce')),
                         _v(r.get('LayerDescription')), _v(r.get('RockType')), _v(r.get('Description'))])
        out[name] = json.dumps(rows, ensure_ascii=False, separators=(',', ':'))
    return out


def logs_from_field(df, name_col):
    """Lithology logs rebuilt from the LOG_FIELD attribute (None if there is none)."""
    if LOG_FIELD not in df.columns:
        return None
    out = []
    for name, txt in zip(df[name_col].astype(str), df[LOG_FIELD]):
        if txt is None or str(txt) in ('', 'NULL', 'nan', 'None'):
            continue
        try:
            rows = json.loads(str(txt))
        except Exception:
            continue
        for r in rows:
            r = list(r) + [None] * (5 - len(r))
            try:
                top, bot = float(r[0]), float(r[1])
            except (TypeError, ValueError):
                continue
            out.append({'BoreholeNo': name, 'Top': top, 'Bottom': bot, 'LayerDescription': r[2],
                        'RockType': r[3], 'Description': r[4]})
    return pd.DataFrame(out, columns=LOG_COLS) if out else None


def attach_log_field(lyr, logs, name_field='name'):
    """Write each borehole's log into the LOG_FIELD attribute of a (file) layer.
    Returns the number of boreholes that got a log."""
    if logs is None or logs.empty or lyr.fields().indexOf(name_field) < 0:
        return 0
    pr = lyr.dataProvider()
    if lyr.fields().indexOf(LOG_FIELD) < 0:
        if not pr.addAttributes([QgsField(LOG_FIELD, FIELD_STRING)]):
            return 0
        lyr.updateFields()
    idx = lyr.fields().indexOf(LOG_FIELD)
    names = [str(f[name_field]) for f in lyr.getFeatures()]
    vals = logs_to_field_values(logs, names)
    changes = {f.id(): {idx: vals[str(f[name_field])]} for f in lyr.getFeatures()
               if str(f[name_field]) in vals}
    if changes:
        pr.changeAttributeValues(changes)
    return len(changes)


import time as _time


def log_timing(what, seconds):
    """Note slow steps in the QGIS log panel (tab 'TEM-Viewer'), to find what takes time."""
    try:
        from qgis.core import QgsMessageLog, Qgis as _Q
        QgsMessageLog.logMessage(f"{what}: {seconds:.1f} s", "TEM-Viewer", _Q.MessageLevel.Info)
    except Exception:
        pass


def _gpkg_points_df(lyr, names):
    """Attributes (+ _FID, _GX, _GY) of a GeoPackage point layer read with sqlite3, or None
    when the layer is not a plain GeoPackage table (filters, unsaved edits ...)."""
    import sqlite3
    import struct
    try:
        if lyr.providerType() != 'ogr' or lyr.subsetString() or lyr.isEditable():
            return None
        parts = lyr.source().split('|')
        path = parts[0]
        if not path.lower().endswith('.gpkg') or not os.path.isfile(path):
            return None
        table = next((p.split('=', 1)[1] for p in parts[1:] if p.lower().startswith('layername=')), None)
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            if table is None:
                rows = con.execute("SELECT table_name FROM gpkg_contents WHERE data_type='features'").fetchall()
                if len(rows) != 1:
                    return None
                table = rows[0][0]
            gcol = con.execute("SELECT column_name FROM gpkg_geometry_columns WHERE table_name=?",
                               (table,)).fetchone()
            info = con.execute(f'PRAGMA table_info("{table}")').fetchall()
            pk = next((r[1] for r in info if r[5]), None)
            if gcol is None or pk is None:
                return None
            gcol = gcol[0]
            cols = [r[1] for r in info]
            missing = [n for n in names if n not in cols]
            if missing:
                return None
            sel = ", ".join(f'"{c}"' for c in names + [pk, gcol])
            data = con.execute(f'SELECT {sel} FROM "{table}"').fetchall()
        finally:
            con.close()
    except Exception:
        return None
    env = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
    gx, gy = np.full(len(data), np.nan), np.full(len(data), np.nan)
    for i, r in enumerate(data):
        b = r[-1]
        if not b or len(b) < 8 or b[:2] != b'GP':
            continue
        flags = b[3]
        o = 8 + env.get((flags >> 1) & 7, 0)
        if len(b) < o + 21:
            continue
        end = '<' if b[o] == 1 else '>'
        gx[i], gy[i] = struct.unpack(end + 'dd', b[o + 5:o + 21])
    n = len(names)
    df = pd.DataFrame([r[:n] for r in data], columns=names)
    df['_FID'] = [r[n] for r in data]
    df['_GX'], df['_GY'] = gx, gy
    return df


def _qtype(series):
    return FIELD_DOUBLE if pd.api.types.is_numeric_dtype(series) else FIELD_STRING


def _layer_from_df(df, x, y, name, crs, columns):
    lyr = QgsVectorLayer(f"Point?crs={crs.authid()}", name, "memory")
    pr = lyr.dataProvider()
    pr.addAttributes([QgsField(str(c), _qtype(df[c])) for c in columns])
    lyr.updateFields()
    feats = []
    for _, r in df.iterrows():
        f = QgsFeature(lyr.fields())
        f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(float(r[x]), float(r[y]))))
        vals = []
        for c in columns:
            v = r[c]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                vals.append(None)
            elif pd.api.types.is_numeric_dtype(df[c]):
                vals.append(float(v))
            else:
                vals.append(str(v))
        f.setAttributes(vals)
        feats.append(f)
    ok, added = pr.addFeatures(feats)
    lyr.updateExtents()
    fids = [f.id() for f in added] if ok and added else list(range(1, len(feats) + 1))
    return lyr, fids


def read_workbook_sheets(path):
    """Every sheet of an .xlsx/.xlsm workbook as a DataFrame (first non-empty row =
    header). Returns {} for other file types or if openpyxl is missing."""
    if os.path.splitext(path or '')[1].lower() not in ('.xlsx', '.xlsm') or not os.path.isfile(path):
        return {}
    try:
        import openpyxl
    except ImportError:
        return {}
    out = {}
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        for ws in wb.worksheets:
            rows = [list(r) for r in ws.iter_rows(values_only=True)
                    if r is not None and any(c is not None for c in r)]
            if not rows:
                continue
            width = max(len(r) for r in rows)
            rows = [r + [None] * (width - len(r)) for r in rows]
            heads, seen = [], {}
            for i, h in enumerate(rows[0]):
                h = str(h).strip() if h is not None and str(h).strip() else f"Col_{i + 1}"
                seen[h] = seen.get(h, 0) + 1
                heads.append(h if seen[h] == 1 else f"{h}_{seen[h]}")
            out[ws.title] = pd.DataFrame(rows[1:], columns=heads)
    finally:
        wb.close()
    return out


def sheet_table_name(sheet_name):
    """GeoPackage table name used for a workbook sheet."""
    return 'sheet_' + re.sub(r'[^0-9A-Za-z_]+', '_', str(sheet_name)).strip('_')[:50]


def _table_layer(df, name):
    """Attribute-only memory table from a DataFrame (numeric columns stay numeric)."""
    tbl = QgsVectorLayer("None", name, "memory")
    kinds = []
    for c in df.columns:
        num = pd.to_numeric(df[c], errors='coerce')
        isnum = df[c].notna().any() and num.notna().sum() == df[c].notna().sum() \
            and not df[c].map(lambda v: isinstance(v, bool)).any()
        kinds.append(isnum)
    names, seen = [], {'fid'}                 # unique ignoring case, as GeoPackage needs
    for c in df.columns:
        n, k = str(c).strip() or "Col", 2
        base = n
        while n.lower() in seen:
            n = f"{base}_{k}"
            k += 1
        seen.add(n.lower())
        names.append(n)
    tbl.dataProvider().addAttributes([QgsField(n, FIELD_DOUBLE if k else FIELD_STRING)
                                      for n, k in zip(names, kinds)])
    tbl.updateFields()
    feats = []
    for row in df.itertuples(index=False):
        f = QgsFeature(tbl.fields())
        vals = []
        for v, k in zip(row, kinds):
            if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NaT:
                vals.append(None)
            elif k:
                vals.append(float(v))
            else:
                vals.append(str(v))
        f.setAttributes(vals)
        feats.append(f)
    tbl.dataProvider().addFeatures(feats)
    return tbl


class DataRegistry(QObject):
    """Links map layers to TEM-Viewer data. Emits `changed` when layers come and go."""
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.soundings = {}   # layer id -> {'df', 'path'}
        self.boreholes = {}   # layer id -> {'bh', 'logs', 'path', 'dem'}
        proj = QgsProject.instance()
        proj.layersRemoved.connect(self._on_removed)
        proj.layersAdded.connect(self._on_added)

    def _on_added(self, *args):
        self.changed.emit()

    def disconnect_project(self):
        proj = QgsProject.instance()
        for sig, slot in ((proj.layersRemoved, self._on_removed), (proj.layersAdded, self._on_added)):
            try:
                sig.disconnect(slot)
            except Exception:
                pass

    def _on_removed(self, ids):
        for i in ids:
            self.soundings.pop(i, None)
            self.boreholes.pop(i, None)
        self.changed.emit()

    # ---------------------------------------------------------------
    #  Creating layers
    # ---------------------------------------------------------------
    def add_soundings(self, df, name, crs, path=""):
        df = standardize_soundings(df)
        lyr, fids = _layer_from_df(df, 'X', 'Y', name, crs, list(df.columns))
        df['_FID'] = fids
        lyr.setCustomProperty(KIND_PROP, 'soundings')
        sym = QgsMarkerSymbol.createSimple({
            'name': 'circle', 'color': '#ff7f00', 'size': '1.8',
            'outline_color': 'black', 'outline_width': '0.2'})
        lyr.setRenderer(QgsSingleSymbolRenderer(sym))
        pal = QgsPalLayerSettings()
        pal.fieldName = 'RECORD'
        pal.enabled = True
        lyr.setLabeling(QgsVectorLayerSimpleLabeling(pal))
        lyr.setLabelsEnabled(True)
        self.soundings[lyr.id()] = {'df': df, 'path': path}
        QgsProject.instance().addMapLayer(lyr)
        return lyr

    def add_boreholes(self, bh, logs, name, crs, path="", dem_layer_id=None, extra_cols=None):
        """extra_cols: optional list of additional summary-sheet columns (e.g. 'TEM')
        that are carried onto the layer as attribute fields."""
        bh = bh.dropna(subset=['XUTM', 'YUTM']).reset_index(drop=True)
        cols = ['BoreholeNo', 'XUTM', 'YUTM', 'DrillDepth', 'Elevation', 'HasLithology']
        view = bh.copy()
        for c in cols:
            if c not in view.columns:
                view[c] = np.nan
        view = view[cols].rename(columns={'BoreholeNo': 'name', 'DrillDepth': 'depth',
                                          'HasLithology': 'has_lithology', 'Elevation': 'elevation',
                                          'XUTM': 'x', 'YUTM': 'y'})
        view['name'] = view['name'].astype(str)
        for c in (extra_cols or []):
            if c not in bh.columns:
                continue
            fname, k = str(c), 2
            # never clobber a built-in field; GeoPackage field names ignore case ('Name' = 'name')
            while fname.lower() in {str(v).lower() for v in view.columns}:
                fname = f"{c}_{k}"
                k += 1
            ser = bh[c]
            num = pd.to_numeric(ser, errors='coerce')
            if ser.notna().any() and num.notna().sum() == ser.notna().sum():
                view[fname] = num.values
            else:
                view[fname] = ser.map(lambda v: None if v is None or
                                      (isinstance(v, float) and np.isnan(v)) else str(v)).values
        if logs is not None and not logs.empty:
            vals = logs_to_field_values(logs, view['name'])
            if vals:
                view[LOG_FIELD] = view['name'].map(lambda n: vals.get(str(n)))
        lyr, fids = _layer_from_df(view, 'x', 'y', name, crs, list(view.columns))
        bh['_FID'] = fids
        lyr.setCustomProperty(KIND_PROP, 'boreholes')
        if logs is not None and not logs.empty:
            logs = logs[logs['BoreholeNo'].astype(str).isin(set(view['name']))].copy()
            try:   # logs travel with the layer (also after reopening TEM-Viewer / the project)
                keep = [c for c in LOG_COLS if c in logs.columns]
                lyr.setCustomProperty(LOGS_PROP, logs[keep].to_json(orient='records'))
            except Exception:
                pass
        self.boreholes[lyr.id()] = {'bh': bh, 'logs': logs if logs is not None else
                                    pd.DataFrame(columns=LOG_COLS), 'path': path, 'dem': dem_layer_id}
        QgsProject.instance().addMapLayer(lyr)
        return lyr

    # ---------------------------------------------------------------
    #  Which layers can be used
    # ---------------------------------------------------------------
    @staticmethod
    def _points(lyr):
        return isinstance(lyr, QgsVectorLayer) and lyr.isValid() and \
            lyr.geometryType() == Qgis.GeometryType.Point

    def is_sounding_layer(self, lyr):
        if not self._points(lyr):
            return False
        if lyr.id() in self.soundings or lyr.customProperty(KIND_PROP) == 'soundings':
            return True
        return lyr.fields().indexOf('RHO_1') >= 0

    def is_borehole_layer(self, lyr):
        if not self._points(lyr) or self.is_sounding_layer(lyr):
            return False
        if lyr.id() in self.boreholes or lyr.customProperty(KIND_PROP) == 'boreholes':
            return True
        names = [f.name() for f in lyr.fields()]
        return _find(names, BH_FIELDS['BoreholeNo']) is not None and \
            _find(names, BH_FIELDS['DrillDepth']) is not None

    def sounding_layers(self):
        return [l for l in QgsProject.instance().mapLayers().values() if self.is_sounding_layer(l)]

    def borehole_layers(self):
        return [l for l in QgsProject.instance().mapLayers().values() if self.is_borehole_layer(l)]

    def is_registered(self, lyr):
        return lyr.id() in self.soundings or lyr.id() in self.boreholes

    # ---------------------------------------------------------------
    #  Data of one layer
    # ---------------------------------------------------------------
    @staticmethod
    def _attributes_df(lyr):
        """All attributes of a point layer + _FID, _GX, _GY. GeoPackages are read straight with
        SQLite (many times faster than feature by feature); other layers through QGIS."""
        names = [f.name() for f in lyr.fields()]
        t0 = _time.perf_counter()
        df = _gpkg_points_df(lyr, names)
        if df is None:
            rows = []
            for f in lyr.getFeatures():
                g = f.geometry()
                p = g.asPoint() if g and not g.isEmpty() else None
                rows.append(list(f.attributes()) + [f.id(), p.x() if p else np.nan, p.y() if p else np.nan])
            df = pd.DataFrame(rows, columns=names + ['_FID', '_GX', '_GY'])
        dt = _time.perf_counter() - t0
        if dt > 0.5:
            log_timing(f"read {len(df)} features of '{lyr.name()}'", dt)
        return df

    def sounding_columns(self, lyr):
        """Column names of a sounding layer as the plugin uses them (no data is read)."""
        if lyr.id() in self.soundings:
            return list(self.soundings[lyr.id()]['df'].columns)
        names = lyr.fields().names()
        ren = canonical_column_names(list(names))
        return [ren.get(n, n) for n in names]

    def soundings_df(self, lyr):
        if lyr.id() in self.soundings:
            return self.soundings[lyr.id()]['df']
        df = self._attributes_df(lyr)
        if df.empty:                      # nothing to cache; layer may still be filling
            return df
        if _find(df.columns, SND_COLS['X']) is None:
            df['X'], df['Y'] = df['_GX'], df['_GY']
        df = standardize_soundings(df)
        self.soundings[lyr.id()] = {'df': df, 'path': lyr.source()}
        return df

    def boreholes_data(self, lyr):
        if lyr.id() in self.boreholes:
            e = self.boreholes[lyr.id()]
            return e['bh'], e['logs']
        df = self._attributes_df(lyr)
        bh = pd.DataFrame({'_FID': df['_FID'], 'XUTM': df['_GX'], 'YUTM': df['_GY']})
        for std, cands in BH_FIELDS.items():
            src = _find(df.columns, cands)
            bh[std] = df[src] if src is not None else np.nan
        bh['BoreholeNo'] = bh['BoreholeNo'].astype(str)
        for c in ('DrillDepth', 'Elevation'):
            bh[c] = pd.to_numeric(bh[c], errors='coerce')
        logs = pd.DataFrame(columns=LOG_COLS)
        stored = lyr.customProperty(LOGS_PROP)
        if not stored:
            glogs = self._logs_from_gpkg(lyr)
            if glogs is None:
                name_src = _find(df.columns, BH_FIELDS['BoreholeNo'])
                if name_src is not None:
                    glogs = logs_from_field(df, name_src)
            if glogs is None and self._gpkg_path(lyr):
                glogs = self._logs_from_sheet_tables(self._gpkg_path(lyr),
                                                     bh['BoreholeNo'].astype(str).tolist())
            if glogs is not None:
                logs = glogs
        if stored:
            try:
                from io import StringIO
                logs = pd.read_json(StringIO(str(stored)), orient='records',
                                     dtype=False, convert_dates=False)   # keep names like '0012' / '86.1230'
                logs['BoreholeNo'] = logs['BoreholeNo'].astype(str)
            except Exception:
                logs = pd.DataFrame(columns=LOG_COLS)
        self.boreholes[lyr.id()] = {'bh': bh, 'logs': logs, 'path': lyr.source(), 'dem': None}
        try:                       # give every lithology a clearly different colour
            from .borehole_section import register_lithologies
            for col in ('RockType', 'LayerDescription'):
                if col in logs.columns:
                    register_lithologies(logs[col].dropna().astype(str))
        except Exception:
            pass
        return bh, logs

    # ---------------------------------------------------------------
    #  Saving a borehole layer (points + lithology logs) to a GeoPackage
    # ---------------------------------------------------------------
    @staticmethod
    def _gpkg_path(lyr):
        src = lyr.source().split('|')[0]
        return src if src.lower().endswith('.gpkg') and os.path.isfile(src) else None

    def _logs_from_gpkg(self, lyr):
        """Lithology logs stored in the 'logs' table next to the borehole layer."""
        path = self._gpkg_path(lyr)
        if not path:
            return None
        tbl = QgsVectorLayer(f"{path}|layername=logs", "logs", "ogr")
        if not tbl.isValid():
            return None
        names = [f.name() for f in tbl.fields()]
        rows = [dict(zip(names, f.attributes())) for f in tbl.getFeatures()]
        if not rows:
            return None
        logs = pd.DataFrame(rows)
        for c in LOG_COLS:
            if c not in logs.columns:
                logs[c] = np.nan
        logs = logs[LOG_COLS].copy()
        logs['BoreholeNo'] = logs['BoreholeNo'].astype(str)
        return logs

    @staticmethod
    def _logs_from_sheet_tables(path, borehole_names):
        """Rebuild lithology logs from the stored per-borehole sheet tables
        (columns by position: Top | Bottom | Description | Lithology)."""
        probe = QgsVectorLayer(path, "probe", "ogr")
        if not probe.isValid():
            return None
        tables = {}
        for sub in probe.dataProvider().subLayers():
            parts = sub.split('!!::!!')
            if len(parts) > 1 and parts[1].startswith('sheet_'):
                tables[parts[1].lower()] = parts[1]
        out = []
        for name in borehole_names:
            tn = tables.get(sheet_table_name(name).lower())
            if tn is None:
                continue
            t = QgsVectorLayer(f"{path}|layername={tn}", tn, "ogr")
            if not t.isValid() or t.fields().count() < 2:
                continue
            for f in t.getFeatures():
                a = f.attributes()
                try:
                    top, bot = float(a[0]), float(a[1])
                except (TypeError, ValueError):
                    continue
                desc = str(a[2]).strip() if len(a) > 2 and a[2] not in (None, '') else ''
                lith = str(a[3]).strip() if len(a) > 3 and a[3] not in (None, '') else ''
                if desc == 'NULL':
                    desc = ''
                if lith == 'NULL':
                    lith = ''
                out.append({'BoreholeNo': str(name), 'Top': min(top, bot), 'Bottom': max(top, bot),
                            'LayerDescription': lith or desc or 'unknown',
                            'RockType': lith or desc or 'unknown', 'Description': desc})
        return pd.DataFrame(out, columns=LOG_COLS) if out else None

    def load_boreholes_gpkg(self, path, name=None, into=None):
        """Add the 'boreholes' layer of a TEM-Viewer GeoPackage and attach its lithology logs
        (the 'logs' table, the lith_log field, or rebuilt from the sheet_* tables).
        into: an existing project layer that should use the GeoPackage instead of its present
        data (it keeps its id, style and place in the Layers panel) – no second layer is added.
        Returns (layer, n_log_rows, source)."""
        uri = f"{path}|layername=boreholes"
        lyr = QgsVectorLayer(uri, name or f"Boreholes - {os.path.splitext(os.path.basename(path))[0]}", "ogr")
        if not lyr.isValid():
            raise RuntimeError("No 'boreholes' layer found in this GeoPackage.")
        if lyr.featureCount() == 0:
            raise RuntimeError("This GeoPackage has no boreholes – it was not saved completely. Import the "
                               "borehole file once more, Plot, and use 'Save borehole layer…' again.")
        logs, src = self._logs_from_gpkg(lyr), "logs table"
        names = [str(f['name']) for f in lyr.getFeatures()] if lyr.fields().indexOf('name') >= 0 else []
        if logs is None and lyr.fields().indexOf(LOG_FIELD) >= 0:
            df = pd.DataFrame([[f['name'], f[LOG_FIELD]] for f in lyr.getFeatures()], columns=['name', LOG_FIELD]) \
                if names else pd.DataFrame()
            logs, src = (logs_from_field(df, 'name') if not df.empty else None), "lith_log field"
        if logs is None:
            logs, src = self._logs_from_sheet_tables(path, names), "sheet tables"
        if logs is None or logs.empty:
            logs, src = None, "none"
        if into is not None:
            src_file = into.customProperty(SOURCE_PROP) or self.source_file(into)
            into.setDataSource(uri, into.name(), "ogr")
            del lyr
            lyr = into
            if src_file:
                lyr.setCustomProperty(SOURCE_PROP, src_file)
            self.boreholes.pop(lyr.id(), None)         # rebuilt from the GeoPackage on first use
        lyr.setCustomProperty(KIND_PROP, 'boreholes')
        if logs is not None:
            lyr.setCustomProperty(LOGS_PROP, logs[LOG_COLS].to_json(orient='records'))
        if into is None:
            QgsProject.instance().addMapLayer(lyr)
        else:
            lyr.triggerRepaint()
            self.changed.emit()
        return lyr, (0 if logs is None else len(logs)), src

    def logs_available(self, lyr):
        """Cheap check (no data read): does this borehole layer have lithology logs somewhere?"""
        if lyr.id() in self.boreholes or lyr.fields().indexOf(LOG_FIELD) >= 0 or lyr.customProperty(LOGS_PROP):
            return True
        path = self._gpkg_path(lyr)
        if path is None:
            return False
        key = (path, os.path.getmtime(path))
        cache = self.__dict__.setdefault('_tables_cache', {})
        if key not in cache:
            cache[key] = [t.lower() for t in self._gpkg_tables(path)]
        return any(t == 'logs' or t.startswith('sheet_') for t in cache[key])

    @staticmethod
    def _gpkg_tables(path):
        probe = QgsVectorLayer(path, "probe", "ogr")
        if not probe.isValid():
            return []
        return [p.split('!!::!!')[1] for p in probe.dataProvider().subLayers() if '!!::!!' in p]

    def source_file(self, lyr):
        """The .bor / Excel file a borehole layer was imported from ('' if unknown)."""
        src = lyr.customProperty(SOURCE_PROP) or ''
        if not src:
            p = str(self.boreholes.get(lyr.id(), {}).get('path') or '')
            if p and not p.split('|')[0].lower().endswith('.gpkg'):
                src = p
        return str(src)

    @staticmethod
    def layers_using_file(path):
        """Project layers whose data source is the file `path`."""
        target = os.path.normcase(os.path.abspath(path))
        return [l for l in QgsProject.instance().mapLayers().values()
                if os.path.normcase(os.path.abspath(l.source().split('|')[0])) == target]

    def save_boreholes(self, lyr, path, sheets=None, replace=()):
        """Save the borehole layer (+ lithology logs + workbook sheets) to a GeoPackage.

        New file: everything is written to a temporary file first, checked (all boreholes there),
        then moved to `path` – a failed save never leaves an empty or half-written GeoPackage.
        Update (`replace` = project layers that already use `path`): the saved layer is updated in
        place through its own connection (no file locking problems); it keeps its id, style and
        place in the Layers panel, so profiles and other settings that point to it keep working."""
        proj = QgsProject.instance()
        bh, logs = self.boreholes_data(lyr)
        logs = logs.copy() if logs is not None else pd.DataFrame(columns=LOG_COLS)
        self._sheet_errors = []
        n_src = int(lyr.featureCount())
        if n_src <= 0:
            raise RuntimeError(f"'{lyr.name()}' has no boreholes – nothing to save. Import the borehole "
                               "file, Plot it, and save that layer.")
        src_file = self.source_file(lyr)
        replace = [r for r in replace if r is not None]
        target = next((r for r in replace if isinstance(r, QgsVectorLayer)
                       and r.customProperty(KIND_PROP) == 'boreholes'), None) or \
            next((r for r in replace if isinstance(r, QgsVectorLayer)
                  and 'layername=boreholes' in r.source().lower()), None)
        if target is not None and target.id() == lyr.id():
            # the layer already is this GeoPackage: only refresh its logs
            self._finish_saved_layer(lyr, logs, src_file)
            lyr._tv_summary = (len(logs), 0)
            return lyr
        if target is not None:
            new = self._update_in_place(lyr, target, logs, sheets, src_file)
        else:
            new = self._write_new_file(lyr, path, logs, sheets, src_file, n_src)
        if proj.mapLayer(lyr.id()) is not None and new.id() != lyr.id():
            proj.removeMapLayer(lyr.id())
        new._tv_sheet_errors = list(self._sheet_errors)
        return new

    # ---- helpers of save_boreholes -------------------------------------
    @staticmethod
    def _writer(path, ctx):
        from qgis.core import QgsVectorFileWriter

        def _enum(nm):
            e = getattr(QgsVectorFileWriter, 'ActionOnExistingFile', None)
            return getattr(e, nm) if e is not None and hasattr(e, nm) \
                else getattr(QgsVectorFileWriter, nm)

        def _write(layer, layer_name, action):
            opt = QgsVectorFileWriter.SaveVectorOptions()
            opt.driverName = 'GPKG'
            opt.layerName = layer_name
            opt.actionOnExistingFile = _enum(action)
            res = QgsVectorFileWriter.writeAsVectorFormatV3(layer, path, ctx, opt)
            code = getattr(res[0], 'value', res[0])
            if code != 0:
                raise RuntimeError(f"Could not write '{layer_name}': {res[1]}")
        return _write

    @staticmethod
    def _logs_table(logs):
        tbl = QgsVectorLayer("None", "logs", "memory")
        tbl.dataProvider().addAttributes([
            QgsField(c, FIELD_DOUBLE if c in ('Top', 'Bottom') else FIELD_STRING) for c in LOG_COLS])
        tbl.updateFields()
        feats = []
        for _, r in logs.iterrows():
            f = QgsFeature(tbl.fields())
            vals = []
            for c in LOG_COLS:
                v = r.get(c)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    vals.append(None)
                elif c in ('Top', 'Bottom'):
                    vals.append(float(v))
                else:
                    vals.append(str(v))
            f.setAttributes(vals)
            feats.append(f)
        tbl.dataProvider().addFeatures(feats)
        return tbl

    def _write_tables(self, write, logs, sheets, first_action):
        """logs + sheet_* tables (the boreholes table is written by the caller)."""
        write(self._logs_table(logs), 'logs', first_action)
        used = {'boreholes', 'logs'}
        for sname, sdf in (sheets or {}).items():
            tname = sheet_table_name(sname)
            base, k = tname, 2
            while tname.lower() in used:
                tname = f"{base}_{k}"
                k += 1
            used.add(tname.lower())
            try:                       # a sheet that cannot be stored must not stop the save
                write(_table_layer(sdf, tname), tname, 'CreateOrOverwriteLayer')
            except Exception as exc:
                self._sheet_errors.append(f"{sname}: {exc}")

    def _finish_saved_layer(self, new, logs, src_file):
        try:                            # logs also inside the borehole features themselves
            attach_log_field(new, logs, 'name' if new.fields().indexOf('name') >= 0 else
                             _find([f.name() for f in new.fields()], BH_FIELDS['BoreholeNo']))
        except Exception:
            pass
        new.setCustomProperty(KIND_PROP, 'boreholes')
        if src_file:
            new.setCustomProperty(SOURCE_PROP, src_file)
        if logs is not None and not logs.empty:
            keep = [c for c in LOG_COLS if c in logs.columns]
            new.setCustomProperty(LOGS_PROP, logs[keep].to_json(orient='records'))
        self.boreholes.pop(new.id(), None)        # rebuilt from the saved layer on first use

    @staticmethod
    def _unique_field_copy(lyr):
        """`lyr`, or an in-memory copy whose field names are unique ignoring case
        ('Name' next to 'name' becomes 'Name_2'), as GeoPackage requires."""
        from qgis.core import QgsFeatureRequest
        seen, renames = set(), {}
        if lyr.providerType() == 'memory':
            seen.add('fid')                   # GeoPackage's own id column
        for i, f in enumerate(lyr.fields()):
            low = f.name().lower()
            if low in seen:
                k, new = 2, f"{f.name()}_2"
                while new.lower() in seen or new.lower() in {x.name().lower() for x in lyr.fields()}:
                    k += 1
                    new = f"{f.name()}_{k}"
                renames[i] = new
                low = new.lower()
            seen.add(low)
        if not renames:
            return lyr
        mem = lyr.materialize(QgsFeatureRequest())
        mem.dataProvider().renameAttributes(renames)
        mem.updateFields()
        return mem

    def _write_new_file(self, lyr, path, logs, sheets, src_file, n_src):
        import time
        from qgis.PyQt.QtCore import QCoreApplication
        proj = QgsProject.instance()
        import shutil
        import tempfile
        folder = os.path.dirname(path) or os.getcwd()
        stem = os.path.splitext(os.path.basename(path))[0]
        # build the file in the local temp folder: cloud folders (OneDrive ...) grab new files
        # while they are being written, and GDAL then cannot reopen them for the next table
        tmp_dir = tempfile.mkdtemp(prefix="tv_")
        tmp = os.path.join(tmp_dir, "boreholes.gpkg")
        write = self._writer(tmp, proj.transformContext())
        write(self._unique_field_copy(lyr), 'boreholes', 'CreateOrOverwriteFile')
        self._write_tables(write, logs, sheets, 'CreateOrOverwriteLayer')
        n_ok = -1
        try:                       # count with sqlite itself: no QGIS handle keeps the file open
            import sqlite3
            con = sqlite3.connect(tmp)
            try:
                n_ok = int(con.execute('SELECT COUNT(*) FROM "boreholes"').fetchone()[0])
            finally:
                con.close()
        except Exception as exc:
            raise RuntimeError(f"The saved file could not be checked: {exc}. Nothing was changed.")
        if n_ok != n_src:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise RuntimeError(f"Saving went wrong: {n_ok} of {n_src} boreholes were written. "
                               "Nothing was changed.")

        def _put(dest):
            for _ in range(20):                       # the target may be busy (OneDrive, viewer)
                try:
                    shutil.copyfile(tmp, dest)
                    return True
                except OSError:
                    QCoreApplication.processEvents()
                    time.sleep(0.25)
            return False
        final = path
        if not _put(final):
            k = 2
            while os.path.exists(os.path.join(folder, f"{stem}_{k}.gpkg")):
                k += 1
            final = os.path.join(folder, f"{stem}_{k}.gpkg")
            if not _put(final):
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise RuntimeError(f"Could not write {path} (is it open in another program?).")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        new = QgsVectorLayer(f"{final}|layername=boreholes", lyr.name(), "ogr")
        if not new.isValid():
            raise RuntimeError("The saved GeoPackage could not be reopened.")
        self._finish_saved_layer(new, logs, src_file)
        try:
            new.setRenderer(lyr.renderer().clone())
            if lyr.labelsEnabled() and lyr.labeling() is not None:
                new.setLabeling(lyr.labeling().clone())
                new.setLabelsEnabled(True)
        except Exception:
            pass
        new._tv_summary = (len(logs), len(sheets or {}))
        new._tv_path = final
        # same place in the Layers panel as the plotted layer
        root = proj.layerTreeRoot()
        node = root.findLayer(lyr.id())
        parent = node.parent() if node is not None else root
        try:
            index = parent.children().index(node) if node is not None else 0
        except ValueError:
            index = 0
        proj.addMapLayer(new, False)
        parent.insertLayer(index, new)
        return new

    def _update_in_place(self, lyr, target, logs, sheets, src_file):
        """Replace the boreholes of a saved GeoPackage layer that is open in the project."""
        from qgis.core import QgsFeatureRequest
        proj = QgsProject.instance()
        if target.isEditable():
            target.rollBack()
        pr = target.dataProvider()
        tr = None
        if lyr.crs().isValid() and target.crs().isValid() and lyr.crs() != target.crs():
            tr = QgsCoordinateTransform(lyr.crs(), target.crs(), proj)
        skip = {'fid', 'ogc_fid'}
        lyr = self._unique_field_copy(lyr)
        have = {f.name().lower() for f in target.fields()}
        missing = [QgsField(f) for f in lyr.fields()
                   if f.name().lower() not in skip and f.name().lower() not in have]
        if missing:
            pr.addAttributes(missing)
            target.updateFields()
        req = QgsFeatureRequest()
        req.setNoAttributes()
        try:
            req.setFlags(QgsFeatureRequest.Flag.NoGeometry)
        except AttributeError:
            req.setFlags(QgsFeatureRequest.NoGeometry)
        old_ids = [f.id() for f in target.getFeatures(req)]
        feats = []
        names = [f.name() for f in lyr.fields() if f.name().lower() not in skip]
        for f in lyr.getFeatures():
            nf = QgsFeature(target.fields())
            g = QgsGeometry(f.geometry())
            if tr is not None and not g.isEmpty():
                g.transform(tr)
            nf.setGeometry(g)
            for nm in names:
                idx = target.fields().lookupField(nm)
                if idx >= 0:
                    nf.setAttribute(idx, f[nm])
            feats.append(nf)
        if old_ids and not pr.deleteFeatures(old_ids):
            raise RuntimeError(f"Could not remove the old boreholes from '{target.name()}' "
                               "(is the file read-only?). Nothing was changed.")
        ok, _added = pr.addFeatures(feats)
        if not ok:
            raise RuntimeError(f"Could not write the new boreholes into '{target.name()}'.")
        path = target.source().split('|')[0]
        write = self._writer(path, proj.transformContext())
        self._update_table(path, 'logs', self._logs_table(logs), write)
        for sname, sdf in (sheets or {}).items():
            tname = sheet_table_name(sname)
            try:
                self._update_table(path, tname, _table_layer(sdf, tname), write)
            except Exception as exc:
                self._sheet_errors.append(f"{sname}: {exc}")
        self._finish_saved_layer(target, logs, src_file)
        target.updateExtents()
        target.reload()
        target.triggerRepaint()
        target._tv_summary = (len(logs), len(sheets or {}))
        target._tv_path = path
        return target

    @staticmethod
    def _update_table(path, table, mem, write):
        """Rows of table `table` in the GeoPackage replaced by those of `mem` (created if missing)."""
        t = QgsVectorLayer(f"{path}|layername={table}", table, "ogr")
        if not t.isValid():
            write(mem, table, 'CreateOrOverwriteLayer')
            return
        pr = t.dataProvider()
        miss = [QgsField(f) for f in mem.fields() if t.fields().indexOf(f.name()) < 0]
        if miss:
            pr.addAttributes(miss)
            t.updateFields()
        ids = [f.id() for f in t.getFeatures()]
        if ids:
            pr.deleteFeatures(ids)
        rows = []
        for f in mem.getFeatures():
            nf = QgsFeature(t.fields())
            for fld in mem.fields():
                idx = t.fields().indexOf(fld.name())
                if idx >= 0:
                    nf.setAttribute(idx, f[fld.name()])
            rows.append(nf)
        pr.addFeatures(rows)

    def dem_layer(self, lyr):
        e = self.boreholes.get(lyr.id())
        return QgsProject.instance().mapLayer(e['dem']) if e and e.get('dem') else None

    # ---------------------------------------------------------------
    #  Combining several layers
    # ---------------------------------------------------------------
    def combine_soundings(self, layers, target_crs=None):
        """Concatenate sounding layers. RECORD stays as in the file unless numbers
        repeat between layers; then layer 2, 3 ... get RECORD + k*base (e.g. 105).
        Adds _LAYER (layer id), _SOURCE (layer name), _SRC_RECORD.
        Coordinates are transformed to target_crs (default: first layer's CRS)."""
        if not layers:
            return None, None
        target = target_crs if (target_crs and target_crs.isValid()) else layers[0].crs()
        frames = []
        for lyr in layers:
            df = self.soundings_df(lyr).copy()
            df['_LAYER'] = lyr.id()
            df['_SOURCE'] = lyr.name()
            df['_SRC_RECORD'] = df['RECORD']
            if lyr.crs().isValid() and target.isValid() and lyr.crs() != target:
                tr = QgsCoordinateTransform(lyr.crs(), target, QgsProject.instance())
                pts = [tr.transform(QgsPointXY(float(x), float(y))) for x, y in zip(df['X'], df['Y'])]
                df['X'] = [p.x() for p in pts]
                df['Y'] = [p.y() for p in pts]
            frames.append(df)
        all_rec = pd.concat([f['RECORD'] for f in frames])
        if len(frames) > 1 and all_rec.duplicated().any():
            base = 10 ** len(str(int(all_rec.max())))
            for k, f in enumerate(frames):
                f['RECORD'] = f['_SRC_RECORD'] + k * base
        out = pd.concat(frames, ignore_index=True, sort=False)
        if out['RECORD'].duplicated().any():
            # RECORD repeats inside a file (e.g. numbering restarts per flight line):
            # give every sounding its own number; the file's value stays in _SRC_RECORD
            out['RECORD'] = np.arange(1, len(out) + 1)
        return out, target

    def combine_boreholes(self, layers, target_crs=None):
        """Concatenate borehole layers (+ logs), coordinates in target_crs."""
        if not layers:
            return None, None
        target = target_crs if (target_crs and target_crs.isValid()) else layers[0].crs()
        bhs, logs = [], []
        for lyr in layers:
            bh, lg = self.boreholes_data(lyr)
            bh = bh.copy()
            bh['_LAYER'] = lyr.id()
            bh['_SOURCE'] = lyr.name()
            if lyr.crs().isValid() and target.isValid() and lyr.crs() != target:
                tr = QgsCoordinateTransform(lyr.crs(), target, QgsProject.instance())
                pts = [tr.transform(QgsPointXY(float(x), float(y))) for x, y in zip(bh['XUTM'], bh['YUTM'])]
                bh['XUTM'] = [p.x() for p in pts]
                bh['YUTM'] = [p.y() for p in pts]
            bhs.append(bh)
            if lg is not None and not lg.empty:
                logs.append(lg)
        bh_all = pd.concat(bhs, ignore_index=True, sort=False)
        logs_all = pd.concat(logs, ignore_index=True, sort=False) if logs else pd.DataFrame(columns=LOG_COLS)
        return (bh_all, logs_all), target


class LayerPicker(QListWidget):
    """Checkable list of the TEM-Viewer-usable layers of one kind ('soundings' or
    'boreholes'). Keeps the user's ticks when layers are added or removed;
    newly created TEM-Viewer layers are ticked automatically."""
    selectionChangedByUser = pyqtSignal()

    def __init__(self, registry, kind, parent=None, max_height=90):
        super().__init__(parent)
        self.registry = registry
        self.kind = kind
        self._known = set()
        self.setMaximumHeight(max_height)
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.itemChanged.connect(lambda *_: self.selectionChangedByUser.emit())
        registry.changed.connect(self.refresh)
        self.refresh()

    def refresh(self):
        checked = set(self.checked_ids())
        layers = (self.registry.sounding_layers() if self.kind == 'soundings'
                  else self.registry.borehole_layers())
        self.blockSignals(True)
        self.clear()
        for lyr in layers:
            label = lyr.name()
            if self.kind == 'boreholes' and not self.registry.is_registered(lyr) and \
                    not self.registry.logs_available(lyr):
                label += "  [no logs]"
            it = QListWidgetItem(label)
            it.setData(Qt.ItemDataRole.UserRole, lyr.id())
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            new = lyr.id() not in self._known
            tick = lyr.id() in checked or (new and (self.registry.is_registered(lyr) or
                                                    bool(lyr.customProperty(KIND_PROP))))
            it.setCheckState(Qt.CheckState.Checked if tick else Qt.CheckState.Unchecked)
            self.addItem(it)
        if self.count() == 0:
            it = QListWidgetItem("(no %s layers – import them first)" % self.kind)
            it.setFlags(Qt.ItemFlag.NoItemFlags)
            self.addItem(it)
        self._known = {l.id() for l in layers}
        self.blockSignals(False)
        # only when the ticked layers really changed: a layer added elsewhere (a saved profile,
        # a map) must not make every tab re-plot
        if set(self.checked_ids()) != checked:
            self.selectionChangedByUser.emit()

    def checked_ids(self):
        out = []
        for i in range(self.count()):
            it = self.item(i)
            if it.flags() & Qt.ItemFlag.ItemIsUserCheckable and it.checkState() == Qt.CheckState.Checked:
                out.append(it.data(Qt.ItemDataRole.UserRole))
        return out

    def set_checked_ids(self, ids):
        """Tick exactly the layers whose ids are given (others unticked)."""
        ids = set(ids)
        self.blockSignals(True)
        for i in range(self.count()):
            it = self.item(i)
            if it.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                it.setCheckState(Qt.CheckState.Checked if it.data(Qt.ItemDataRole.UserRole) in ids
                                 else Qt.CheckState.Unchecked)
        self.blockSignals(False)
        self.selectionChangedByUser.emit()

    def checked_layers(self):
        proj = QgsProject.instance()
        return [l for l in (proj.mapLayer(i) for i in self.checked_ids()) if l is not None]
