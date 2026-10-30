# -*- coding: utf-8 -*-
"""
TEM-Viewer - link between the map and the "TEM-Viewer Logs" dock.

Boreholes and soundings are shown side by side (same depth scale) when they are
  * selected on the map (any borehole or sounding layer TEM-Viewer can use), or
  * clicked with QGIS' own Identify Features tool (nearest borehole and/or
    sounding within the search radius; Ctrl/Shift+click adds to the columns).
"""
import numpy as np
import pandas as pd

from qgis.PyQt.QtCore import Qt, QObject, QEvent
from qgis.core import (QgsProject, QgsCoordinateTransform, QgsRectangle, QgsFeatureRequest,
                       QgsPointXY, QgsGeometry, Qgis)
from qgis.gui import QgsMapTool, QgsMapToolIdentify

from .lithology_dock import LithologyDockWidget
from .resistivity_tool import (PARAMS, PARAM_ORDER, available_params, layer_geometry, layer_values,
                               get_colormap)

MAX_COLUMNS = 8


def nearest_features(registry, pt, radius, crs, layers):
    """Nearest borehole and nearest sounding to pt (in crs) within radius:
    [('bh' | 'snd', layer id, feature id), ...]."""
    best = {}
    for lyr in layers:
        kind = 'snd' if registry.is_sounding_layer(lyr) else 'bh'
        p, r = QgsPointXY(pt), radius
        rect = QgsRectangle(p.x() - r, p.y() - r, p.x() + r, p.y() + r)
        if lyr.crs().isValid() and crs.isValid() and lyr.crs() != crs:
            tr = QgsCoordinateTransform(crs, lyr.crs(), QgsProject.instance())
            rect = tr.transformBoundingBox(rect)
            p = tr.transform(p)
        g = QgsGeometry.fromPointXY(p)
        for f in lyr.getFeatures(QgsFeatureRequest().setFilterRect(rect)):
            fg = f.geometry()
            if fg is None or fg.isEmpty():
                continue
            d = fg.distance(g)
            if kind not in best or d < best[kind][0]:
                best[kind] = (d, (kind, lyr.id(), int(f.id())))
    return [best[k][1] for k in ('bh', 'snd') if k in best]


class LogLink(QObject):
    def __init__(self, iface, registry, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.registry = registry
        self.items = []             # [('bh' | 'snd', layer_id, fid)]
        self._layers = []           # layers whose selectionChanged we listen to
        self._from_selection = False
        self.dock = LithologyDockWidget(iface.mainWindow())
        iface.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)
        self.dock.hide()
        self.dock.paramChanged.connect(self.render)
        self.dock.clearRequested.connect(self.clear)
        self.dock.set_hint("Select boreholes and/or soundings on the map, or click them with the "
                           "Identify Features tool (Ctrl/Shift+click adds more).")
        self._vp = iface.mapCanvas().viewport()
        self._vp.installEventFilter(self)
        registry.changed.connect(self._rewire)
        self._rewire()
        from .borehole_section import color_events
        color_events().changed.connect(self._colors_changed)

    # ------------------------------------------------------------------
    #  Layer selections
    # ------------------------------------------------------------------
    def _usable_layers(self):
        return self.registry.borehole_layers() + self.registry.sounding_layers()

    def _rewire(self, *args):
        for lyr in self._layers:
            try:
                lyr.selectionChanged.disconnect(self._on_selection)
            except Exception:
                pass
        self._layers = self._usable_layers()
        for lyr in self._layers:
            lyr.selectionChanged.connect(self._on_selection)
        alive = {l.id() for l in self._layers}
        n = len(self.items)
        self.items = [it for it in self.items if it[1] in alive]
        if len(self.items) != n:
            self.render()

    def _on_selection(self, *args):
        items, too_many = [], []
        for lyr in self._layers:
            try:
                ids = lyr.selectedFeatureIds()
            except RuntimeError:
                continue
            if not ids:
                continue
            kind = 'snd' if self.registry.is_sounding_layer(lyr) else 'bh'
            if len(ids) > MAX_COLUMNS:
                too_many.append(f"{len(ids)} selected in '{lyr.name()}'")
                continue
            items += [(kind, lyr.id(), int(f)) for f in sorted(ids)]
        if not items:
            if too_many and self.dock.isVisible():
                self.dock.info_label.setText("Not shown (more than %d): %s." % (MAX_COLUMNS, "; ".join(too_many)))
            return                      # nothing to show: leave the panel as it is
        self.items = items[:MAX_COLUMNS]
        note = ""
        if too_many:
            note = "Not shown (more than %d): %s." % (MAX_COLUMNS, "; ".join(too_many))
        self.render(note=note)

    # ------------------------------------------------------------------
    #  Identify Features tool
    # ------------------------------------------------------------------
    def eventFilter(self, obj, ev):
        try:
            if (ev.type() == QEvent.Type.MouseButtonRelease and ev.button() == Qt.MouseButton.LeftButton
                    and isinstance(self.iface.mapCanvas().mapTool(), QgsMapToolIdentify)):
                add = bool(ev.modifiers() & (Qt.KeyboardModifier.ControlModifier |
                                             Qt.KeyboardModifier.ShiftModifier))
                pos = ev.position().toPoint() if hasattr(ev, 'position') else ev.pos()
                self.identify_at(pos.x(), pos.y(), add)
        except Exception:
            pass
        return False                    # QGIS' own Identify keeps working

    def identify_at(self, px, py, add=False):
        canvas = self.iface.mapCanvas()
        pt = canvas.getCoordinateTransform().toMapCoordinates(int(px), int(py))
        radius = QgsMapTool.searchRadiusMU(canvas)
        visible = {l.id() for l in canvas.layers()}
        found = self.nearest(pt, radius, canvas.mapSettings().destinationCrs(),
                             [l for l in self._layers if l.id() in visible])
        if not found:
            return
        if add:
            items = list(self.items)
            for it in found:
                if it not in items:
                    items.append(it)
            self.items = items[-MAX_COLUMNS:]
        else:
            self.items = found
        self.render()

    def nearest(self, pt, radius, crs, layers):
        """Nearest borehole and nearest sounding to pt (map CRS) within radius."""
        return nearest_features(self.registry, pt, radius, crs, layers)

    # ------------------------------------------------------------------
    #  Columns
    # ------------------------------------------------------------------
    def clear(self):
        self.items = []
        self.dock.setWindowTitle("TEM-Viewer Logs")
        self.dock.info_label.setText("")
        self.dock.set_params([], {})
        self.dock.set_hint("Select boreholes and/or soundings on the map, or click them with the "
                           "Identify Features tool (Ctrl/Shift+click adds more).")

    def _borehole_column(self, lyr, fid):
        bh, logs = self.registry.boreholes_data(lyr)
        row = bh[bh['_FID'] == fid]
        if row.empty:
            return None, "borehole not found"
        name = str(row['BoreholeNo'].iloc[0])
        sub = logs[logs['BoreholeNo'].astype(str) == name] if logs is not None and not logs.empty else None
        if sub is None or sub.empty:              # shown as an empty column, ready for a future log
            depth = pd.to_numeric(row['DrillDepth'].iloc[0], errors='coerce') if 'DrillDepth' in row else np.nan
            return {'name': name, 'subtitle': lyr.name() + "  ·  no lithology log yet", 'intervals': [],
                    'max_bottom': float(depth) if pd.notna(depth) and depth > 0 else 10.0}, None
        has_desc = 'Description' in sub.columns
        intervals = []
        for _, r in sub.sort_values('Top').iterrows():
            general = str(r.get('RockType') or r.get('LayerDescription') or '')
            detail = str(r.get('Description') or '') if has_desc else ''
            if not detail or detail == 'nan':
                detail = str(r.get('LayerDescription') or '')
            intervals.append({'top': float(r['Top']), 'bottom': float(r['Bottom']),
                              'detail': detail, 'general': general})
        return {'name': name, 'subtitle': lyr.name(), 'intervals': intervals,
                'max_bottom': max(i['bottom'] for i in intervals)}, None

    def _colour_scale(self, lyr, prm):
        """Colour range of a parameter: RES 1–1000 Ohm·m, IP from the layer's data."""
        if prm == 'RHO':
            return 1.0, 1000.0, True
        v = layer_values(self.registry.soundings_df(lyr), prm).ravel()
        log = prm != 'C'
        v = v[np.isfinite(v) & ((v > 0) if log else True)]
        if not v.size:
            return (1.0, 1000.0, True) if log else (0.0, 1.0, False)
        lo, hi = np.percentile(v, [2, 98])
        if hi <= lo:
            hi = lo * 10 if log else lo + 1
        return float(lo), float(hi), log

    def _sounding_column(self, lyr, fid, prm):
        df = self.registry.soundings_df(lyr)
        row = df[df['_FID'] == fid]
        if row.empty:
            return None, "sounding not found"
        if prm not in available_params(row):
            prm = 'RHO'
        top, bot = layer_geometry(row)
        vals = layer_values(row, prm)[0, :top.shape[1]]
        info = PARAMS[prm]
        doi = np.nan
        for c in (info['doi'][0], PARAMS['RHO']['doi'][0]):
            if c in row.columns:
                d = pd.to_numeric(row[c], errors='coerce').iloc[0]
                if pd.notna(d) and d > 0:
                    doi = float(d)
                    break
        top, bot = top[0], bot[0]
        if np.isfinite(doi):
            show = doi * 1.3
        else:
            show = float(np.nanmax(top)) * 1.05 + 5
        show = min(show, float(np.nanmax(bot)))
        lo, hi, log = self._colour_scale(lyr, prm)
        rec = row['RECORD'].iloc[0]
        elev = pd.to_numeric(row['ELEVATION'], errors='coerce').iloc[0]
        sub = lyr.name() + (f"  ·  {elev:.1f} m a.s.l." if pd.notna(elev) else "")
        snd = dict(top=top, bot=bot, vals=vals, doi=doi, log=log, vmin=lo, vmax=hi,
                   cmap=get_colormap('BlueToPink'), unit=info['unit'],
                   label='ρ' if prm == 'RHO' else info['label'])
        return {'kind': 'sounding', 'name': f"Sounding {int(rec)}", 'subtitle': sub, 'snd': snd,
                'max_bottom': show}, None

    def render(self, *args, note=""):
        proj = QgsProject.instance()
        # parameters available among the shown soundings
        params = set()
        for kind, lid, fid in self.items:
            lyr = proj.mapLayer(lid)
            if kind == 'snd' and lyr is not None:
                try:
                    params.update(available_params(pd.DataFrame(columns=self.registry.sounding_columns(lyr))))
                except Exception:
                    pass
        plist = [p for p in PARAM_ORDER if p in params]
        self.dock.set_params(plist, {p: PARAMS[p]['label'] for p in plist})
        prm = self.dock.current_param()
        columns, errors = [], []
        for kind, lid, fid in self.items:
            lyr = proj.mapLayer(lid)
            if lyr is None:
                continue
            try:
                col, err = (self._sounding_column(lyr, fid, prm) if kind == 'snd'
                            else self._borehole_column(lyr, fid))
            except Exception as exc:
                col, err = None, str(exc)
            if col is not None:
                columns.append(col)
            elif err:
                errors.append(err)
        if not self.dock.isVisible():
            self.dock.show()
        self.dock.raise_()
        msg = "; ".join([m for m in (note, "; ".join(errors)) if m])
        self.dock.info_label.setText(msg)
        if not columns:
            self.dock.setWindowTitle("TEM-Viewer Logs")
            self.dock.set_hint("Nothing to show for this selection." if self.items or note else
                               "Select boreholes and/or soundings on the map, or click them with the "
                               "Identify Features tool (Ctrl/Shift+click adds more).")
            return
        depth = max(c['max_bottom'] for c in columns)
        depth = depth + max(2.0, 0.03 * depth)
        self.dock.show_columns(columns, depth)
        nb = sum(1 for c in columns if c.get('kind') != 'sounding')
        ns = len(columns) - nb
        parts = ([f"{nb} borehole{'s' if nb > 1 else ''}"] if nb else []) + \
                ([f"{ns} sounding{'s' if ns > 1 else ''}"] if ns else [])
        self.dock.setWindowTitle("TEM-Viewer Logs – " + ", ".join(parts))

    def show_hint_for(self, layer):
        """After plotting boreholes: point the user to selection / Identify."""
        if not self.dock.isVisible():
            self.dock.show()
        self.dock.raise_()
        try:
            self.iface.messageBar().pushMessage(
                "TEM-Viewer", f"'{layer.name()}': select boreholes or click them with Identify Features "
                           "to see their logs (soundings work the same way).",
                level=Qgis.MessageLevel.Info, duration=8)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _colors_changed(self):
        if self.items:
            self.render()

    def cleanup(self):
        try:
            from .borehole_section import color_events
            color_events().changed.disconnect(self._colors_changed)
        except Exception:
            pass
        for lyr in self._layers:
            try:
                lyr.selectionChanged.disconnect(self._on_selection)
            except Exception:
                pass
        self._layers = []
        try:
            self.registry.changed.disconnect(self._rewire)
        except Exception:
            pass
        try:
            self._vp.removeEventFilter(self)
        except Exception:
            pass
        if self.dock is not None:
            try:
                self.iface.removeDockWidget(self.dock)
                self.dock.deleteLater()
            except Exception:
                pass
            self.dock = None
