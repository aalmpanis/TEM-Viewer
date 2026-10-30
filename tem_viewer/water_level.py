# -*- coding: utf-8 -*-
"""
TEM-Viewer - water levels of boreholes for the Profile tab.

The water level is read from a column the user chooses. Columns come from every table
that belongs to a borehole layer:
  attr          the layer's attribute table
  bh            the borehole table imported this session (all summary-sheet columns)
  sheet:<name>  a sheet of the source Excel workbook, or a sheet_* table of the GeoPackage
The row of a borehole is the one whose name cell matches the borehole name; the column
holding the names is found automatically (most matches).
"""
import os
import re

import numpy as np
import pandas as pd

from qgis.core import QgsVectorLayer

_WB_CACHE = {}          # (path, mtime) -> {sheet: DataFrame}
WL_HINTS = ('water', 'vand', 'pejl', 'spejl', 'gwl', 'wl', 'level', 'grundv')


def norm_name(v):
    """Comparable form of a borehole name ('86. 1230', ' 86.1230 ' -> '86.1230'; 12.0 -> '12')."""
    if v is None:
        return ''
    if isinstance(v, float):
        if not np.isfinite(v):
            return ''
        if v.is_integer():
            v = int(v)
    s = re.sub(r'\s+', '', str(v)).lower()
    if s in ('nan', 'null', 'none'):
        return ''
    if re.fullmatch(r'\d+\.0', s):
        s = s[:-2]
    return s


def to_number(v):
    if v is None:
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        return float(v)
    s = str(v).strip().replace(',', '.')
    m = re.search(r'-?\d+(\.\d+)?', s)
    try:
        return float(m.group(0)) if m else np.nan
    except ValueError:
        return np.nan


def key_of(src, col):
    return f"{src}\t{col}"


def split_key(key):
    src, _, col = str(key).partition('\t')
    return src, col


def label_of(src, col):
    if src == 'attr':
        return f"{col}   (layer attributes)"
    if src == 'bh':
        return f"{col}   (borehole table)"
    return f"{col}   (sheet '{src.split(':', 1)[1]}')"


def looks_like_wl(key):
    col = split_key(key)[1].lower()
    return any(h in col for h in WL_HINTS)


def detect_name_col(df, names, prefer=('name', 'boreholeno', 'borehole', 'dgu_nr', 'dgunr', 'id')):
    """Column of df whose values match most of the borehole names (None if none match)."""
    best, best_n = None, 0
    low = {str(c).lower(): c for c in df.columns}
    order = [low[p] for p in prefer if p in low] + [c for c in df.columns if str(c).lower() not in prefer]
    for c in order:
        try:
            n = int(df[c].map(norm_name).isin(names).sum())
        except Exception:
            continue
        if n > best_n:
            best, best_n = c, n
    return best


def _layer_df(lyr, with_geometry=False):
    names = [f.name() for f in lyr.fields()]
    rows = [list(f.attributes()) for f in lyr.getFeatures()]
    return pd.DataFrame(rows, columns=names)


def _workbook(path):
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return {}
    if key not in _WB_CACHE:
        from .tv_data import read_workbook_sheets
        _WB_CACHE[key] = read_workbook_sheets(path)
    return _WB_CACHE[key]


def borehole_tables(registry, lyr):
    """{source: {'df': DataFrame, 'name_col': column with the borehole names}}."""
    from .tv_data import sheet_table_name
    bh, _logs = registry.boreholes_data(lyr)
    names = set(bh['BoreholeNo'].map(norm_name)) - {''} if bh is not None and not bh.empty else set()
    out = {}

    attr = _layer_df(lyr).drop(columns=['lith_log', 'fid'], errors='ignore')
    nc = detect_name_col(attr, names)
    if nc is not None:
        out['attr'] = dict(df=attr, name_col=nc)

    if bh is not None and not bh.empty:
        extra = [c for c in bh.columns if c not in attr.columns and not str(c).startswith('_')
                 and c not in ('XUTM', 'YUTM', 'BoreholeNo', 'DrillDepth', 'Elevation', 'HasLithology')]
        if extra:
            out['bh'] = dict(df=bh[['BoreholeNo'] + extra].copy(), name_col='BoreholeNo')

    # sheets of the source workbook (imported this session) or of the GeoPackage
    entry = getattr(registry, 'boreholes', {}).get(lyr.id(), {}) or {}
    path = str(entry.get('path') or '').split('|')[0]
    sheets = {}
    if path.lower().endswith(('.xlsx', '.xlsm')) and os.path.isfile(path):
        sheets = {sn: df for sn, df in _workbook(path).items() if norm_name(sn) not in names}
    else:
        gpkg = registry._gpkg_path(lyr) if hasattr(registry, '_gpkg_path') else None
        if gpkg:
            per_bh = {sheet_table_name(n).lower() for n in bh['BoreholeNo'].astype(str)} if names else set()
            probe = QgsVectorLayer(gpkg, "probe", "ogr")
            subs = probe.dataProvider().subLayers() if probe.isValid() else []
            for sub in subs:
                parts = sub.split('!!::!!')
                if len(parts) < 2:
                    continue
                tn = parts[1]
                if not tn.lower().startswith('sheet_') or tn.lower() in per_bh:
                    continue
                t = QgsVectorLayer(f"{gpkg}|layername={tn}", tn, "ogr")
                if t.isValid():
                    sheets[tn[6:]] = _layer_df(t)
    for sn, df in sheets.items():
        if df is None or df.empty:
            continue
        nc = detect_name_col(df, names)
        if nc is not None:
            out[f"sheet:{sn}"] = dict(df=df, name_col=nc)
    return out


def values_for(tables, key):
    """{normalised borehole name: water level} for the chosen column, None if this layer
    has no such column."""
    src, col = split_key(key)
    tbl = tables.get(src)
    if tbl is None or col not in tbl['df'].columns:
        return None
    df, nc = tbl['df'], tbl['name_col']
    out = {}
    for name, v in zip(df[nc].map(norm_name), df[col].map(to_number)):
        if name and np.isfinite(v) and name not in out:
            out[name] = float(v)
    return out


FIELD_ID_NAMES = ('field_id', 'fieldid', 'field id', 'field-id', 'field_no', 'field_nr', 'fieldname',
                  'field_name', 'field')


def field_ids(tables):
    """{normalised borehole name: Field_id text} from the first table that has a Field_id
    column (layer attributes first, then the borehole table, then the sheets)."""
    order = ['attr', 'bh'] + sorted(k for k in tables if k.startswith('sheet:'))
    for src in order:
        tbl = tables.get(src)
        if tbl is None:
            continue
        low = {str(c).lower().strip(): c for c in tbl['df'].columns}
        col = next((low[n] for n in FIELD_ID_NAMES if n in low), None)
        if col is None:
            continue
        out = {}
        for name, v in zip(tbl['df'][tbl['name_col']].map(norm_name), tbl['df'][col]):
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                continue
            txt = str(int(v)) if isinstance(v, float) and v.is_integer() else str(v).strip()
            if name and txt and txt not in ('NULL', 'nan') and name not in out:
                out[name] = txt
        if out:
            return out
    return {}
