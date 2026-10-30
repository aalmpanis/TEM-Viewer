# -*- coding: utf-8 -*-
"""
TEM-Viewer - A4 report pages: location map + profile section.

Everything that draws or stores a report page lives here, so the "Graph + location
map - PDF" export of the Profile tab.
Only the background-map rendering needs QGIS (imported lazily), the rest is plain
matplotlib and can be tested without QGIS.

  prepare_location_map()   map extent + background rendered from the QGIS canvas
  draw_location_map()      square map (Easting / Northing axes) as PNG bytes
  figure_to_png()          re-render a live profile figure at a given size / font size
  compose_page()           one A4 page: title, map, info block, profile, footer
  export_report_pdf()      many pages -> one PDF
  save_report_file() / load_report_file()   self-contained, re-editable report (.btreport)
  item_to_json() / item_from_json()         a saved profile (images included) as text

A "report item" (one saved profile) is a dict:
  uid, name, start, end, length_m, n_snd, search_w, layers, crs, created, yaxis, params,
  map:     {'landscape': png bytes, 'portrait': png bytes}
  profile: {'landscape': png bytes, 'portrait': png bytes}
"""
import io
import os
import re
import math
import json
import uuid
import base64
import zipfile
import tempfile
import datetime

import numpy as np
import matplotlib
import matplotlib.image as mpimg
import matplotlib.text
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_agg import FigureCanvasAgg

A4 = {'landscape': (11.69, 8.27), 'portrait': (8.27, 11.69)}
ORIENTATIONS = ('landscape', 'portrait')
A4_LANDSCAPE = A4['landscape']          # kept for older callers

MARGIN = 0.4            # page margin (in)
TITLE_H = 0.5           # title band
FOOTER_H = 0.3          # footer band
MAP_SIDE = {'landscape': 2.5, 'portrait': 3.6}      # side of the square map area (in)
MAP_LAB_L, MAP_LAB_B = 0.95, 0.65                   # room for Northing / Easting labels (in)
MAP_PAD = 0.15

# font sizes (pt) used when a profile figure is re-rendered for the page
FS_AXIS_LABEL = 11
FS_TICK = 10
FS_TEXT_MIN = 8.5
FS_TEXT_MAX = 13

DEFAULT_SETTINGS = dict(
    orientation='landscape', cover=False, title='Resistivity profiles', subtitle='',
    author='', date='', footer='', numbers=True, contents=True)


def new_uid():
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
#  Geometry helpers
# ---------------------------------------------------------------------------
def _to_crs(xy, src, dst):
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if src is None or dst is None or not src.isValid() or not dst.isValid() or src == dst:
        return xy
    from qgis.core import QgsProject, QgsCoordinateTransform, QgsPointXY
    tr = QgsCoordinateTransform(src, dst, QgsProject.instance())
    out = [tr.transform(QgsPointXY(float(x), float(y))) for x, y in xy]
    return np.array([[p.x(), p.y()] for p in out], dtype=float)


def _nice(v):
    if v <= 0:
        return 1.0
    e = 10 ** math.floor(math.log10(v))
    for m in (1, 2, 5, 10):
        if m * e >= v:
            return m * e
    return 10 * e


def render_canvas_map(iface, extent, width_px, height_px, dpi=200):
    """Render the canvas layers for extent (x0, y0, x1, y1 in the canvas CRS).
    Returns (rgb(a) array, (x0, x1, y0, y1) of the rendered extent) or (None, None)."""
    from qgis.PyQt.QtCore import QSize
    from qgis.core import QgsMapSettings, QgsMapRendererParallelJob, QgsRectangle
    canvas = iface.mapCanvas()
    ms = QgsMapSettings(canvas.mapSettings())
    ms.setOutputSize(QSize(int(width_px), int(height_px)))
    ms.setOutputDpi(dpi)
    ms.setExtent(QgsRectangle(*extent))
    job = QgsMapRendererParallelJob(ms)
    job.start()
    job.waitForFinished()
    img = job.renderedImage()
    vis = ms.visibleExtent()
    fd, tmp = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        if not img.save(tmp, "PNG"):
            return None, None
        arr = mpimg.imread(tmp)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    return arr, (vis.xMinimum(), vis.xMaximum(), vis.yMinimum(), vis.yMaximum())


# ---------------------------------------------------------------------------
#  Page layout (inches, origin bottom-left)
# ---------------------------------------------------------------------------
def page_layout(orientation='landscape'):
    """Rectangles (x, y, w, h in inches) of the parts of a profile page."""
    W, H = A4[orientation]
    side = MAP_SIDE[orientation]
    map_w = side + MAP_LAB_L + MAP_PAD
    map_h = side + MAP_LAB_B + MAP_PAD
    top = H - MARGIN - TITLE_H
    map_rect = (MARGIN, top - map_h, map_w, map_h)
    info_x = MARGIN + map_w + 0.3
    info_rect = (info_x, top - map_h, W - MARGIN - info_x, map_h)
    prof_y = MARGIN + FOOTER_H
    prof_h = (top - map_h) - 0.2 - prof_y
    prof_rect = (MARGIN, prof_y, W - 2 * MARGIN, prof_h)
    return dict(size=(W, H), map=map_rect, info=info_rect, profile=prof_rect,
                title_y=H - MARGIN, footer_y=MARGIN)


# ---------------------------------------------------------------------------
#  Location map
# ---------------------------------------------------------------------------
def prepare_location_map(iface, line_xy, line_crs, soundings_xy=None, boreholes=None,
                         renderer=None, px=1100):
    """Square map extent around the profile (+ soundings, boreholes) and its background
    rendered from the QGIS canvas (all layers shown there, incl. any basemap).
    boreholes: list of dicts with x, y, name (in line_crs). renderer(extent, w, h) can
    replace the QGIS rendering (tests). Returns a dict for draw_location_map()."""
    canvas_crs = iface.mapCanvas().mapSettings().destinationCrs() if iface is not None else None
    crs = canvas_crs if (canvas_crs is not None and canvas_crs.isValid()) else line_crs
    line = _to_crs(line_xy, line_crs, crs)
    snd = (_to_crs(soundings_xy, line_crs, crs)
           if soundings_xy is not None and len(soundings_xy) else None)
    bh_xy = (_to_crs([[b['x'], b['y']] for b in boreholes], line_crs, crs) if boreholes else None)
    bh_names = [str(b.get('name', '')) for b in boreholes] if boreholes else []
    geographic = bool(crs is not None and crs.isValid() and crs.isGeographic())

    pts = [line] + ([snd] if snd is not None else []) + ([bh_xy] if bh_xy is not None else [])
    allp = np.vstack(pts)
    allp = allp[np.isfinite(allp).all(axis=1)]
    x0, y0 = allp.min(axis=0)
    x1, y1 = allp.max(axis=0)
    span = max(x1 - x0, y1 - y0, 1e-9)
    pad = max(0.12 * span, 0.0005 if geographic else 50.0)
    side = span + 2 * pad
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ext = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)

    img, vis, note = None, None, ""
    try:
        if renderer is not None:
            img, vis = renderer(ext, px, px)
        elif iface is not None:
            img, vis = render_canvas_map(iface, ext, px, px)
    except Exception as exc:
        note = f"(map could not be rendered: {exc})"
    if img is None and not note:
        note = "(no map layers rendered)"
    return dict(crs_id=(crs.authid() if crs is not None and crs.isValid() else ""),
                geographic=geographic, line=line, snd=snd, bh=bh_xy, bh_names=bh_names,
                ext=ext, img=img, vis=vis, note=note)


def _decorate_map(ax, geographic, crs_id, north_scale=1.0):
    """North arrow, scale bar, Easting / Northing (or Lon / Lat) axes and the CRS.
    north_scale shrinks the north arrow (0.5 = half size)."""
    # north arrow + scale bar
    tip = 1.0 - 0.06 * max(north_scale, 0.3)
    ax.annotate('N', xy=(0.94, tip), xytext=(0.94, tip - 0.14 * north_scale), xycoords='axes fraction',
                ha='center', va='center', fontsize=max(10 * north_scale, 6.5), fontweight='bold',
                arrowprops=dict(arrowstyle='-|>', color='black', lw=max(1.5 * north_scale, 0.9),
                                mutation_scale=max(10 * north_scale, 6)), zorder=9)
    if not geographic:
        xl, yl = ax.get_xlim(), ax.get_ylim()
        L = _nice((xl[1] - xl[0]) / 5.0)
        sx, sy = xl[0] + 0.04 * (xl[1] - xl[0]), yl[0] + 0.05 * (yl[1] - yl[0])
        ax.plot([sx, sx + L], [sy, sy], color='black', linewidth=3, zorder=9, solid_capstyle='butt')
        ax.plot([sx, sx + L], [sy, sy], color='white', linewidth=1, zorder=9, solid_capstyle='butt')
        lab = f"{L / 1000:g} km" if L >= 1000 else f"{L:g} m"
        ax.text(sx + L / 2, sy + 0.015 * (yl[1] - yl[0]), lab, ha='center', va='bottom', fontsize=8,
                zorder=9, bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.7))

    # Easting / Northing axes
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.tick_params(labelsize=8, length=3)
    for lab in ax.get_yticklabels():
        lab.set_rotation(90)
        lab.set_va('center')
    if geographic:
        ax.set_xlabel("Longitude (°)", fontsize=9, labelpad=3)
        ax.set_ylabel("Latitude (°)", fontsize=9, labelpad=3)
    else:
        ax.set_xlabel("Easting (m)", fontsize=9, labelpad=3)
        ax.set_ylabel("Northing (m)", fontsize=9, labelpad=3)
    if crs_id:
        ax.text(1.0, 1.01, crs_id, transform=ax.transAxes, ha='right', va='bottom',
                fontsize=7, color='0.4')


def draw_location_map(prep, side_in=None, start_end=("A", "B"), dpi=220, orientation='landscape'):
    """Square location map with Easting / Northing axes. Returns PNG bytes of a figure
    of (side + label margins) inches, ready to be placed on a page."""
    side = side_in if side_in else MAP_SIDE[orientation]
    fw, fh = side + MAP_LAB_L + MAP_PAD, side + MAP_LAB_B + MAP_PAD
    fig = Figure(figsize=(fw, fh))
    FigureCanvasAgg(fig)
    ax = fig.add_axes([MAP_LAB_L / fw, MAP_LAB_B / fh, side / fw, side / fh])
    line, snd, bh_xy = prep['line'], prep['snd'], prep['bh']
    ext, vis, img = prep['ext'], prep['vis'], prep['img']

    if img is not None:
        ax.imshow(img, extent=vis, origin='upper', interpolation='bilinear', zorder=0)
        ax.set_xlim(vis[0], vis[1])
        ax.set_ylim(vis[2], vis[3])
    else:
        ax.set_xlim(ext[0], ext[2])
        ax.set_ylim(ext[1], ext[3])
        ax.set_facecolor('#f4f4f4')
        if prep['note']:
            ax.text(0.5, 0.5, prep['note'], transform=ax.transAxes, ha='center', va='center',
                    color='0.4', fontsize=8, wrap=True)
    ax.set_aspect('equal', adjustable='box')

    ax.plot(line[:, 0], line[:, 1], color='white', linewidth=4.5, zorder=3, solid_capstyle='round')
    ax.plot(line[:, 0], line[:, 1], color='#d62728', linewidth=2.2, zorder=4, solid_capstyle='round')
    if snd is not None:
        ax.scatter(snd[:, 0], snd[:, 1], s=6, c='black', zorder=5, linewidths=0)
    if bh_xy is not None:
        ax.scatter(bh_xy[:, 0], bh_xy[:, 1], s=40, marker='^', c='#ffd400', edgecolors='black',
                   linewidths=0.8, zorder=6)
        for (bx, by), nm in zip(bh_xy, prep['bh_names']):
            ax.annotate(nm, (bx, by), xytext=(4, 4), textcoords='offset points', fontsize=7,
                        zorder=7, bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
    for (px_, py_), lab in ((line[0], start_end[0]), (line[-1], start_end[1])):
        ax.annotate(lab, (px_, py_), xytext=(0, 0), textcoords='offset points', ha='center',
                    va='center', fontsize=10, fontweight='bold', zorder=8,
                    bbox=dict(boxstyle='circle,pad=0.25', fc='white', ec='#d62728', lw=1.2))
    # direction arrow in the middle of the line
    seg = np.diff(line, axis=0)
    cum = np.concatenate([[0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
    if len(seg) and cum[-1] > 0:
        m = cum[-1] / 2
        k = min(max(int(np.searchsorted(cum, m) - 1), 0), len(seg) - 1)
        t = (m - cum[k]) / max(cum[k + 1] - cum[k], 1e-12)
        mid = line[k] + t * seg[k]
        d = seg[k] / max(np.hypot(*seg[k]), 1e-12)
        L = 0.04 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        ax.annotate('', xy=(mid[0] + d[0] * L, mid[1] + d[1] * L),
                    xytext=(mid[0] - d[0] * L, mid[1] - d[1] * L),
                    arrowprops=dict(arrowstyle='-|>', color='#d62728', lw=2), zorder=8)

    _decorate_map(ax, prep['geographic'], prep['crs_id'])
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, facecolor='white')
    return buf.getvalue()


# ---------------------------------------------------------------------------
#  Overview map page (chosen layers: profiles, boreholes, soundings, background)
# ---------------------------------------------------------------------------
OV_LEGEND_H = 0.85          # legend band under the overview map (in)
PROFILE_COLORS = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#17becf', '#8c564b',
                  '#e377c2', '#bcbd22', '#7f7f7f']
BH_STYLES = [('#ffd400', '^'), ('#ff7f0e', 's'), ('#2ca02c', 'D'), ('#e377c2', 'v'), ('#17becf', 'p')]
SND_COLORS = ['#000000', '#555555', '#1f3a93', '#6b2d5c', '#2d6b3a']


def overview_layout(orientation='landscape'):
    """Rect (x, y, w, h in inches) of the overview image on a page, and the axes
    size (w, h) inside that image."""
    W, H = A4[orientation]
    top = H - MARGIN - TITLE_H
    y0 = MARGIN + FOOTER_H
    rect = (MARGIN, y0, W - 2 * MARGIN, top - 0.1 - y0)
    ax_w = rect[2] - MAP_LAB_L - MAP_PAD
    ax_h = rect[3] - MAP_LAB_B - OV_LEGEND_H - MAP_PAD - 0.1
    return rect, (ax_w, ax_h)


def fit_extent(bbox, aspect, pad_frac=0.06, min_pad=0.0):
    """Extent (x0, y0, x1, y1) around bbox with padding, widened to width / height = aspect."""
    x0, y0, x1, y1 = [float(v) for v in bbox]
    span = max(x1 - x0, y1 - y0, 1e-9)
    pad = max(pad_frac * span, min_pad)
    x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
    w, h = x1 - x0, y1 - y0
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def render_layers_map(layers, crs, extent, width_px, height_px, dpi=200):
    """Render exactly `layers` (top layer first, with their QGIS styles) for extent in crs.
    Returns (image array, (x0, x1, y0, y1)) or (None, None)."""
    from qgis.PyQt.QtCore import QSize
    from qgis.PyQt.QtGui import QColor
    from qgis.core import QgsMapSettings, QgsMapRendererParallelJob, QgsRectangle, QgsProject
    if not layers:
        return None, None
    ms = QgsMapSettings()
    ms.setLayers(layers)
    ms.setDestinationCrs(crs)
    ms.setTransformContext(QgsProject.instance().transformContext())
    ms.setBackgroundColor(QColor(255, 255, 255))
    ms.setOutputSize(QSize(int(width_px), int(height_px)))
    ms.setOutputDpi(dpi)
    ms.setExtent(QgsRectangle(*extent))
    job = QgsMapRendererParallelJob(ms)
    job.start()
    job.waitForFinished()
    img = job.renderedImage()
    vis = ms.visibleExtent()
    fd, tmp = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        if not img.save(tmp, "PNG"):
            return None, None
        arr = mpimg.imread(tmp)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    return arr, (vis.xMinimum(), vis.xMaximum(), vis.yMinimum(), vis.yMaximum())


def overview_bbox(data):
    """Bounding box of the drawn profiles / boreholes / soundings (None if nothing)."""
    arrs = [np.asarray(part, float) for p in data.get('profiles', []) for part in p['parts']]
    arrs += [np.asarray(b['xy'], float) for b in data.get('bh', []) if len(b['xy'])]
    arrs += [np.asarray(s['xy'], float) for s in data.get('snd', []) if len(s['xy'])]
    arrs = [a.reshape(-1, 2) for a in arrs if a.size]
    if not arrs:
        return None
    a = np.vstack(arrs)
    a = a[np.isfinite(a).all(axis=1)]
    if not len(a):
        return None
    return (a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max())


def draw_overview_map(data, size_in, extent, img=None, vis=None, dpi=220, orientation='landscape'):
    """Overview map with legend as PNG bytes. data: dict with
      profiles: [{name, parts: [Nx2 arrays]}]   (drawn as coloured lines, optional labels)
      bh:       [{layer, xy: Nx2, names: [..]}]  (triangles etc., optional labels)
      snd:      [{layer, xy: Nx2}]               (small dots)
      background: [layer names] rendered by QGIS into img
      label_profiles, label_bh, crs_id, geographic, note."""
    fw, fh = size_in
    fig = Figure(figsize=(fw, fh))
    FigureCanvasAgg(fig)
    ax_w = fw - MAP_LAB_L - MAP_PAD
    ax_h = fh - MAP_LAB_B - OV_LEGEND_H - MAP_PAD - 0.1
    ax = fig.add_axes([MAP_LAB_L / fw, (MAP_LAB_B + OV_LEGEND_H) / fh, ax_w / fw, ax_h / fh])
    if img is not None and vis is not None:
        ax.imshow(img, extent=vis, origin='upper', interpolation='bilinear', zorder=0)
        ax.set_xlim(vis[0], vis[1])
        ax.set_ylim(vis[2], vis[3])
    else:
        ax.set_xlim(extent[0], extent[2])
        ax.set_ylim(extent[1], extent[3])
        ax.set_facecolor('#f7f7f7')
    ax.set_aspect('equal', adjustable='box')
    handles = []

    for k, s in enumerate(data.get('snd', [])):
        xy = np.asarray(s['xy'], float).reshape(-1, 2)
        if not len(xy):
            continue
        c = SND_COLORS[k % len(SND_COLORS)]
        ax.scatter(xy[:, 0], xy[:, 1], s=4, c=c, zorder=3, linewidths=0)
        handles.append(Line2D([], [], color=c, marker='o', ls='', ms=3, label=f"Soundings: {s['layer']}"))

    profiles = data.get('profiles', [])
    for k, p in enumerate(profiles):
        c = PROFILE_COLORS[k % len(PROFILE_COLORS)]
        longest = None
        for part in p['parts']:
            a = np.asarray(part, float).reshape(-1, 2)
            if len(a) < 2:
                continue
            ax.plot(a[:, 0], a[:, 1], color='white', linewidth=4.0, zorder=4, solid_capstyle='round')
            ax.plot(a[:, 0], a[:, 1], color=c, linewidth=2.0, zorder=5, solid_capstyle='round')
            ax.plot(a[0, 0], a[0, 1], marker='o', ms=4.5, mfc='white', mec=c, mew=1.3, zorder=6)
            if longest is None or len(a) > len(longest):
                longest = a
        if longest is not None and data.get('label_profiles', True):
            seg = np.diff(longest, axis=0)
            cum = np.concatenate([[0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
            m = cum[-1] / 2
            j = min(max(int(np.searchsorted(cum, m) - 1), 0), len(seg) - 1)
            t = (m - cum[j]) / max(cum[j + 1] - cum[j], 1e-12)
            mid = longest[j] + t * seg[j]
            ax.annotate(p['name'], (mid[0], mid[1]), xytext=(5, 5), textcoords='offset points', fontsize=7.5,
                        color=c, fontweight='bold', zorder=8,
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=c, lw=0.6, alpha=0.85))
        if k < 12:
            handles.append(Line2D([], [], color=c, lw=2.0, label=p['name']))
    if len(profiles) > 12:
        handles.append(Line2D([], [], color='0.5', lw=2.0, label=f"+ {len(profiles) - 12} more profiles"))

    for k, b in enumerate(data.get('bh', [])):
        xy = np.asarray(b['xy'], float).reshape(-1, 2)
        if not len(xy):
            continue
        c, mk = BH_STYLES[k % len(BH_STYLES)]
        ax.scatter(xy[:, 0], xy[:, 1], s=34, marker=mk, c=c, edgecolors='black', linewidths=0.7, zorder=6)
        if data.get('label_bh', False):
            for (bx, by), nm in zip(xy, b.get('names', [])):
                if nm:
                    ax.annotate(str(nm), (bx, by), xytext=(3, 3), textcoords='offset points', fontsize=6,
                                zorder=7, bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.75))
        handles.append(Line2D([], [], color=c, marker=mk, mec='black', ls='', ms=6,
                              label=f"Boreholes: {b['layer']}"))

    if img is None and data.get('note'):
        ax.text(0.5, 0.02, data['note'], transform=ax.transAxes, ha='center', va='bottom',
                color='0.45', fontsize=7)
    _decorate_map(ax, data.get('geographic', False), data.get('crs_id', ''), north_scale=0.5)

    # legend band under the map
    lax = fig.add_axes([MAP_LAB_L / fw, 0.05 / fh, ax_w / fw, (OV_LEGEND_H - 0.1) / fh])
    lax.set_axis_off()
    if handles:
        ncol = 3 if orientation == 'landscape' else 2
        lax.legend(handles=handles, loc='upper left', ncol=ncol, fontsize=8, frameon=False,
                   handletextpad=0.4, columnspacing=1.5, borderaxespad=0)
    bg = data.get('background') or []
    if bg:
        txt = "Background: " + ", ".join(bg)
        if len(txt) > 160:
            txt = txt[:157] + "…"
        lax.text(0.0, 0.0, txt, transform=lax.transAxes, ha='left', va='bottom', fontsize=7, color='0.4')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, facecolor='white')
    return buf.getvalue()


def build_overview_item(name, data, extent_mode='fit', canvas_extent=None, renderer=None,
                        orientations=None, uid=None, ov=None):
    """Report item of kind 'overview': one map image per page format.
    extent_mode: 'fit' (drawn features), 'all' (canvas_extent = union of the ticked layers)
    or 'view' (canvas_extent = current map view); renderer(extent, w_px, h_px) -> (img, vis)
    renders the background layers. ov: the dialog settings (stored to edit it later)."""
    orientations = orientations or ORIENTATIONS
    item = make_item(name, '', '', kind='overview')
    if uid:
        item['uid'] = uid
    item['ov'] = ov or {}
    bbox = overview_bbox(data) if extent_mode == 'fit' else None
    if bbox is None:
        bbox = canvas_extent
    if bbox is None:
        raise ValueError("Nothing to show: tick at least one profile, borehole or sounding layer, "
                         "or choose another map extent.")
    geographic = data.get('geographic', False)
    pad = 0.0 if extent_mode == 'view' else 0.06
    min_pad = 0.0 if extent_mode == 'view' else (0.002 if geographic else 150.0)
    notes = []
    for o in orientations:
        rect, (ax_w, ax_h) = overview_layout(o)
        ext = fit_extent(bbox, ax_w / ax_h, pad, min_pad)
        img, vis = None, None
        if renderer is not None and data.get('background'):
            try:
                img, vis = renderer(ext, int(ax_w * 200), int(ax_h * 200))
            except Exception as exc:
                notes.append(f"(background could not be rendered: {exc})")
        d = dict(data)
        if notes:
            d['note'] = notes[-1]
        item['map'][o] = draw_overview_map(d, rect[2:], ext, img, vis, orientation=o)
    item['n_profiles'] = len(data.get('profiles', []))
    item['n_bh'] = int(sum(len(b['xy']) for b in data.get('bh', [])))
    return item


def compose_overview_page(fig, item, orientation='landscape', caption=None, footer='', page_no=None,
                          total=None, title_size=15):
    fig.clear()
    W, H = A4[orientation]
    L = page_layout(orientation)
    fig.set_size_inches(W, H, forward=False)
    fig.set_facecolor('white')
    fig.text(0.5, L['title_y'] / H, caption or item.get('name', ''), ha='center', va='top',
             fontsize=title_size, fontweight='bold')
    rect, _ = overview_layout(orientation)
    ax = fig.add_axes(_frac(rect, W, H))
    ax.set_axis_off()
    arr = _decode(item, 'map', orientation)
    if arr is not None:
        ax.imshow(arr, interpolation='antialiased', aspect='equal')
        ax.set_anchor('N')
    else:
        ax.text(0.5, 0.5, "No overview map stored for this page format\n(edit the overview map to render it)",
                ha='center', va='center', transform=ax.transAxes, color='0.4')
    _footer(fig, L, footer, page_no, total)
    return fig


# ---------------------------------------------------------------------------
#  Horizontal colorbar under each panel
# ---------------------------------------------------------------------------
CB_RATIO = 0.07


def panel_gridspec(fig, n_panels, with_colorbar=True):
    """GridSpec with, for every panel, a row for the plot and (optionally) a thin row
    below it for its horizontal colorbar. Use panel_axes_spec(k) / colorbar_spec(k)."""
    if not with_colorbar:
        return fig.add_gridspec(n_panels, 1)
    return fig.add_gridspec(2 * n_panels, 3, height_ratios=[1, CB_RATIO] * n_panels,
                            width_ratios=[0.3, 0.4, 0.3])


def horizontal_colorbar(fig, gs, k, mappable, label):
    """Colorbar of panel k in the thin row under it (40 % of the panel width, centred)."""
    cax = fig.add_subplot(gs[2 * k + 1, 1])
    cax.set_label('<colorbar>')
    return fig.colorbar(mappable, cax=cax, orientation='horizontal', label=label)


# ---------------------------------------------------------------------------
#  Profile figure -> PNG at the size of its slot on the page
# ---------------------------------------------------------------------------
def _manual_stack(fig, mains, cbs, size_in):
    """Place several panels (each with a horizontal colorbar below) in inches, so tall
    decorations never squeeze the plots in a short A4 slot."""
    W, H = size_in
    left, right, top = 0.85, 0.3, 0.25
    cb_h, gap, cb_text, x_text = 0.1, 0.08, 0.5, 0.5
    for cax in cbs:                              # compact colorbar text
        cax.tick_params(axis='x', labelsize=8)
        cax.xaxis.label.set_fontsize(9)
    n = len(mains)
    pads = [gap + cb_h + cb_text + 0.08 + (x_text if k == n - 1 else 0.0) for k in range(n)]
    ph = max((H - top - sum(pads)) / n, 0.35)
    w = W - left - right
    y_top = H - top
    for k, (ax, cax) in enumerate(zip(mains, cbs)):
        y0 = y_top - ph
        ax.set_position([left / W, y0 / H, w / W, ph / H])
        cy = y0 - (x_text if k == n - 1 else 0.0) - gap - cb_h
        cax.set_position([(left + 0.3 * w) / W, cy / H, 0.4 * w / W, cb_h / H])
        y_top = cy - cb_text - 0.08


def figure_to_png(fig, size_in, dpi=200, before=None):
    """Render a live profile figure at size_in (w, h inches) with larger fonts, then put
    the figure back exactly as it was. Axis labels / ticks get FS_AXIS_LABEL / FS_TICK pt,
    other texts are enlarged. before() is called first (e.g. to hide the plot cursor)."""
    if before is not None:
        before()
    old_size = fig.get_size_inches().copy()
    texts = list(fig.findobj(matplotlib.text.Text))
    tick_ids, axis_ids = set(), set()
    tick_old = []
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            labs = axis.get_ticklabels()
            tick_ids.update(id(t) for t in labs)
            axis_ids.add(id(axis.label))
            tick_old.append((ax, axis.axis_name, labs[0].get_fontsize() if labs else None))
    sizes = [(t, t.get_fontsize()) for t in texts]
    positions = [(ax, ax.get_position().frozen()) for ax in fig.axes]
    sub = {k: getattr(fig.subplotpars, k) for k in ('left', 'right', 'bottom', 'top', 'wspace', 'hspace')}
    try:
        fig.set_size_inches(size_in[0], size_in[1], forward=False)
        for t, sz in sizes:
            if id(t) in tick_ids:
                continue
            if id(t) in axis_ids:
                t.set_fontsize(FS_AXIS_LABEL)
            elif t.get_text():
                t.set_fontsize(min(max(sz * 1.25, FS_TEXT_MIN), FS_TEXT_MAX))
        import warnings
        axes_lab = [t for t, _ in sizes if id(t) in axis_ids and t.get_text()]
        for scale in (1.0, 0.85, 0.72, 0.6):
            for ax in fig.axes:
                ax.tick_params(axis='both', labelsize=FS_TICK * scale)
            for t in axes_lab:
                t.set_fontsize(FS_AXIS_LABEL * scale)
            ok = True
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('error')
                    fig.tight_layout(h_pad=0.4)
                main = [a for a in fig.axes if a.get_label() != '<colorbar>']
                if main and min(a.get_position().height for a in main) < 0.5 / max(len(main), 1) \
                        and scale > 0.6:
                    ok = False                 # plots squeezed by the colorbar decorations
            except Exception:
                ok = False
            if ok:
                break
        mains = [a for a in fig.axes if a.get_label() != '<colorbar>']
        cbs = [a for a in fig.axes if a.get_label() == '<colorbar>']
        if len(mains) > 1 and len(cbs) == len(mains):
            _manual_stack(fig, mains, cbs, size_in)
        # colorbar titles must fit the colorbar's length
        fw, fh = size_in
        for ax in fig.axes:
            if ax.get_label() == '<colorbar>':
                pos = ax.get_position()
                w_in, h_in = pos.width * fw, pos.height * fh
                lab, length = (ax.xaxis.label, w_in) if w_in >= h_in else (ax.yaxis.label, h_in)
                n = max(len(lab.get_text()), 1)
                lab.set_fontsize(max(6.5, min(FS_AXIS_LABEL, 0.85 * 72 * length / (0.62 * n))))
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, facecolor='white')
        return buf.getvalue()
    finally:
        fig.set_size_inches(old_size[0], old_size[1], forward=False)
        for t, sz in sizes:
            t.set_fontsize(sz)
        for ax, name, sz in tick_old:
            if sz is not None:
                ax.tick_params(axis=name, labelsize=sz)
        try:                                   # put the layout back exactly as it was
            fig.subplots_adjust(**sub)
        except Exception:
            pass
        for ax, pos in positions:
            ax.set_position(pos)


# ---------------------------------------------------------------------------
#  Report items
# ---------------------------------------------------------------------------
def make_item(name, start, end, **meta):
    item = dict(uid=new_uid(), name=name, start=start, end=end, length_m=0.0, n_snd=0,
                search_w=0.0, layers='', crs='', yaxis='', params='', kind='profile',
                created=datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                map={}, profile={})
    item.update(meta)
    return item


def build_report_item(fig, name, line_xy, line_crs, iface, start_end, soundings_xy=None,
                      boreholes=None, bh_dist=None, orientations=None, before=None,
                      renderer=None, kind='profile', **meta):
    """Render everything a report page needs from a live figure: the location map and the
    figure itself for each page format. boreholes: dicts with x, y, name, offset (as drawn
    on the figure); bh_dist: the 'boreholes within' distance, None if boreholes are off.
    meta: length_m, n_snd, search_w, layers, crs, yaxis, params."""
    orientations = orientations or ORIENTATIONS
    st, en = start_end
    item = make_item(name, st, en, kind=kind, **meta)
    if bh_dist:
        offs = [float(b['offset']) for b in (boreholes or []) if b.get('offset') is not None]
        item['bh_dist'] = float(bh_dist)
        item['n_bh'] = len(boreholes or [])
        item['bh_off'] = [min(offs), max(offs)] if offs else None
        item['bh_list'] = [[str(b.get('name', '')), float(b['offset'])] for b in (boreholes or [])
                           if b.get('offset') is not None]
    prep = prepare_location_map(iface, np.asarray(line_xy, float), line_crs, soundings_xy, boreholes,
                                renderer=renderer)
    for o in orientations:
        item['map'][o] = draw_location_map(prep, MAP_SIDE[o], (st, en), orientation=o)
        item['profile'][o] = figure_to_png(fig, page_layout(o)['profile'][2:], before=before)
    return item


def item_ready(item, orientation=None):
    orients = [orientation] if orientation else ORIENTATIONS
    if item.get('kind') == 'overview':
        return all(item.get('map', {}).get(o) for o in orients)
    return all(item.get('map', {}).get(o) and item.get('profile', {}).get(o) for o in orients)


def item_to_json(item):
    d = {k: v for k, v in item.items() if k not in ('map', 'profile') and not k.startswith('_')}
    d['v'] = 1
    d['map'] = {o: base64.b64encode(b).decode('ascii') for o, b in item.get('map', {}).items() if b}
    d['profile'] = {o: base64.b64encode(b).decode('ascii') for o, b in item.get('profile', {}).items() if b}
    return json.dumps(d)


def item_from_json(text):
    """Saved profile from the text stored in a profile layer. None if it is not valid."""
    try:
        d = json.loads(text)
        if not isinstance(d, dict) or 'uid' not in d:
            return None
        d['map'] = {o: base64.b64decode(s) for o, s in d.get('map', {}).items()}
        d['profile'] = {o: base64.b64decode(s) for o, s in d.get('profile', {}).items()}
        return d
    except Exception:
        return None


def _decode(item, kind, orientation):
    """Image array of an item part (cached on the item)."""
    cache = item.setdefault('_cache', {})
    key = (kind, orientation)
    if key not in cache:
        data = item.get(kind, {}).get(orientation)
        cache[key] = mpimg.imread(io.BytesIO(data), format='png') if data else None
    return cache[key]


# ---------------------------------------------------------------------------
#  Page composition
# ---------------------------------------------------------------------------
def _frac(r, W, H):
    x, y, w, h = r
    return [x / W, y / H, w / W, h / H]


def compose_page(fig, item, orientation='landscape', caption=None, footer='', page_no=None,
                 total=None, title_size=15):
    """Draw one profile page (title, location map, info block, profile) onto fig."""
    if item.get('kind') == 'overview':
        return compose_overview_page(fig, item, orientation, caption, footer, page_no, total, title_size)
    fig.clear()
    L = page_layout(orientation)
    W, H = L['size']
    fig.set_size_inches(W, H, forward=False)
    fig.set_facecolor('white')

    fig.text(0.5, L['title_y'] / H, caption or item.get('name', ''), ha='center', va='top',
             fontsize=title_size, fontweight='bold')

    # location map
    ax_map = fig.add_axes(_frac(L['map'], W, H))
    ax_map.set_axis_off()
    arr = _decode(item, 'map', orientation)
    if arr is not None:
        ax_map.imshow(arr, interpolation='antialiased', aspect='equal')
    else:
        ax_map.text(0.5, 0.5, "No location map stored\n(save the profile again)", ha='center',
                    va='center', transform=ax_map.transAxes, color='0.4')
        ax_map.set_axis_on()
        ax_map.set_xticks([])
        ax_map.set_yticks([])

    # info block
    ax_i = fig.add_axes(_frac(L['info'], W, H))
    ax_i.set_axis_off()
    ax_i.set_xlim(0, 1)
    ax_i.set_ylim(0, 1)
    rows = []
    if item.get('length_m'):
        rows.append(("Length", f"{item['length_m'] / 1000:.2f} km" if item['length_m'] >= 1000
                     else f"{item['length_m']:.0f} m"))
    if item.get('n_snd'):
        rows.append(("Soundings", f"{item['n_snd']}"))
    if item.get('search_w'):
        rows.append(("Search width", f"{item['search_w']:g} m"))
    if item.get('bh_dist'):
        bl = item.get('bh_list')
        if bl:
            bl = sorted(bl, key=lambda r: r[1])
            if len(bl) == 1:
                txt = f"{bl[0][1]:.0f} m ({bl[0][0]})" if bl[0][0] else f"{bl[0][1]:.0f} m"
            else:
                txt = ", ".join(f"{n}: {d:.0f} m" if n else f"{d:.0f} m" for n, d in bl[:8])
                if len(bl) > 8:
                    txt += f" … (+{len(bl) - 8} more)"
        elif item.get('n_bh') and item.get('bh_off'):          # older items
            lo, hi = item['bh_off']
            txt = f"{lo:.0f}–{hi:.0f} m"
        else:
            txt = "no borehole found"
        rows.append(("Borehole distance", txt))
    info_w, info_h = L['info'][2], L['info'][3]
    narrow = info_w < 4.0
    val_x = (0.52 if narrow else 0.30)
    y_in = info_h - 0.03                      # work in inches from the top of the block
    for key, val in rows:
        if narrow and key == "Borehole distance":
            key = "Borehole dist."
        lines = _wrap(val, 18 if narrow else 40)
        ax_i.text(0.0, y_in / info_h, key, ha='left', va='top', fontsize=10, color='0.35', fontweight='bold')
        ax_i.text(val_x, y_in / info_h, "\n".join(lines), ha='left', va='top', fontsize=10, linespacing=1.3)
        y_in -= 0.21 * len(lines) + 0.12
    handles = [Line2D([], [], color='#d62728', lw=2.2, label='Profile line'),
               Line2D([], [], color='black', marker='o', ls='', ms=3, label='Soundings'),
               Line2D([], [], color='#ffd400', marker='^', mec='black', ls='', ms=7, label='Boreholes')]
    ax_i.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.0, max(y_in - 0.1, 0.5) / info_h),
                fontsize=9, frameon=False, ncol=1 if narrow else 3, handletextpad=0.4,
                columnspacing=1.2, borderaxespad=0)

    # profile section, fitted into its slot (top-centred)
    ax_p = fig.add_axes(_frac(L['profile'], W, H))
    ax_p.set_axis_off()
    arr = _decode(item, 'profile', orientation)
    if arr is not None:
        ax_p.imshow(arr, interpolation='antialiased', aspect='equal')
        ax_p.set_anchor('N')
    else:
        ax_p.text(0.5, 0.5, "No profile image stored for this page format\n(save the profile again)",
                  ha='center', va='center', transform=ax_p.transAxes, color='0.4')
    _footer(fig, L, footer, page_no, total)
    return fig


def _wrap(text, width):
    words, lines, cur = str(text).replace(';', ',').split(), [], ''
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + ' ' + w).strip()
    if cur:
        lines.append(cur)
    return lines[:4] or ['']


def _footer(fig, L, footer, page_no, total):
    W, H = L['size']
    y = L['footer_y'] / H
    if footer:
        fig.text(MARGIN / W, y, footer, ha='left', va='bottom', fontsize=8, color='0.35')
    if page_no is not None:
        txt = f"Page {page_no}" + (f" / {total}" if total else "")
        fig.text(1 - MARGIN / W, y, txt, ha='right', va='bottom', fontsize=8, color='0.35')


def compose_cover(fig, settings, entries, orientation='landscape'):
    """Cover page: title, subtitle, author / date and (optionally) the contents.
    entries: [(caption, page number)]"""
    fig.clear()
    L = page_layout(orientation)
    W, H = L['size']
    fig.set_size_inches(W, H, forward=False)
    fig.set_facecolor('white')
    y = 0.70 if orientation == 'landscape' else 0.78
    fig.text(0.5, y, settings.get('title') or '', ha='center', va='center', fontsize=28, fontweight='bold')
    if settings.get('subtitle'):
        fig.text(0.5, y - 0.07, settings['subtitle'], ha='center', va='center', fontsize=16, color='0.3')
    meta = "   ·   ".join(x for x in (settings.get('author'), settings.get('date')) if x)
    if meta:
        fig.text(0.5, y - 0.14, meta, ha='center', va='center', fontsize=12, color='0.4')
    if settings.get('contents') and entries:
        yy = y - 0.24
        fig.text(0.5, yy, "Contents", ha='center', va='center', fontsize=13, fontweight='bold')
        yy -= 0.045
        step = min(0.032, (yy - 0.08) / max(len(entries), 1))
        for cap, no in entries:
            fig.text(0.30, yy, cap, ha='left', va='center', fontsize=10)
            fig.text(0.70, yy, str(no), ha='right', va='center', fontsize=10)
            yy -= step
    return fig


# ---------------------------------------------------------------------------
#  Whole report
# ---------------------------------------------------------------------------
def _usable_pages(pages, items):
    return [p for p in pages if p.get('uid') in items and item_ready(items[p['uid']])]


def page_count(pages, items, settings):
    return len(_usable_pages(pages, items)) + (1 if settings.get('cover') else 0)


def render_report_page(fig, index, pages, items, settings):
    """Draw page `index` (0-based, cover first when enabled) of the report onto fig.
    Returns the number of pages. Pages whose profile is missing are skipped."""
    orient = settings.get('orientation', 'landscape')
    use = _usable_pages(pages, items)
    cover = bool(settings.get('cover'))
    total = len(use) + (1 if cover else 0)
    first = 2 if cover else 1
    if total == 0:
        fig.clear()
        W, H = A4[orient]
        fig.set_size_inches(W, H, forward=False)
        fig.set_facecolor('white')
        fig.text(0.5, 0.5, "No pages yet.\nAdd saved profiles to the report.", ha='center',
                 va='center', fontsize=14, color='0.5')
        return 0
    index = min(max(index, 0), total - 1)
    if cover and index == 0:
        entries = [((p.get('caption') or items[p['uid']]['name']), first + i) for i, p in enumerate(use)]
        compose_cover(fig, settings, entries, orient)
        return total
    p = use[index - (1 if cover else 0)]
    compose_page(fig, items[p['uid']], orient, caption=p.get('caption') or None,
                 footer=settings.get('footer', ''),
                 page_no=(index + 1) if settings.get('numbers') else None, total=total)
    return total


def new_page_figure():
    fig = Figure(figsize=A4['landscape'])
    FigureCanvasAgg(fig)
    return fig


def render_report_png(index, pages, items, settings, dpi=130):
    """One report page as PNG bytes (for the live preview). Returns (png, n_pages)."""
    fig = new_page_figure()
    n = render_report_page(fig, index, pages, items, settings)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, facecolor='white')
    return buf.getvalue(), n


def export_report_pdf(path, pages, items, settings, progress=None):
    """Write the whole report as one A4 PDF. Returns the number of pages."""
    from matplotlib.backends.backend_pdf import PdfPages
    n = page_count(pages, items, settings)
    if n == 0:
        raise ValueError("The report has no pages.")
    meta = {'Title': settings.get('title') or 'TEM-Viewer report', 'Creator': 'TEM-Viewer'}
    if settings.get('author'):
        meta['Author'] = settings['author']
    with PdfPages(path, metadata=meta) as pdf:
        for i in range(n):
            fig = new_page_figure()
            render_report_page(fig, i, pages, items, settings)
            pdf.savefig(fig, facecolor='white')
            if progress:
                progress(i + 1, n)
    return n


def save_item_pdf(path, item, orientation='landscape', caption=None):
    """One profile page as a PDF (the 'Graph + location map - PDF' export)."""
    from matplotlib.backends.backend_pdf import PdfPages
    fig = new_page_figure()
    compose_page(fig, item, orientation, caption=caption)
    with PdfPages(path, metadata={'Title': item.get('name', ''), 'Creator': 'TEM-Viewer'}) as pdf:
        pdf.savefig(fig, facecolor='white')
    return path


# ---------------------------------------------------------------------------
#  Report file (.btreport = zip with report.json + images): re-importable any time
# ---------------------------------------------------------------------------
REPORT_EXT = '.btreport'


def save_report_file(path, pages, items, settings):
    """Store settings, page order, captions and the images of every used profile."""
    used = [p for p in pages if p.get('uid') in items]
    doc = dict(version=1, settings={k: settings.get(k, v) for k, v in DEFAULT_SETTINGS.items()},
               pages=[dict(uid=p['uid'], caption=p.get('caption', '')) for p in used], items={})
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in used:
            it = items[p['uid']]
            if p['uid'] in doc['items']:
                continue
            meta = {k: v for k, v in it.items() if k not in ('map', 'profile') and not k.startswith('_')}
            meta['images'] = {}
            for kind in ('map', 'profile'):
                for o, data in it.get(kind, {}).items():
                    if data:
                        name = f"images/{it['uid']}_{kind}_{o}.png"
                        z.writestr(name, data)
                        meta['images'][f"{kind}_{o}"] = name
            doc['items'][it['uid']] = meta
        z.writestr('report.json', json.dumps(doc, indent=1))
    return path


def load_report_file(path):
    """Returns (settings, pages, items) from a .btreport file."""
    with zipfile.ZipFile(path) as z:
        doc = json.loads(z.read('report.json').decode('utf-8'))
        items = {}
        for uid, meta in doc.get('items', {}).items():
            it = {k: v for k, v in meta.items() if k != 'images'}
            it['map'], it['profile'] = {}, {}
            for key, name in meta.get('images', {}).items():
                kind, o = key.split('_', 1)
                it[kind][o] = z.read(name)
            items[uid] = it
    settings = dict(DEFAULT_SETTINGS)
    settings.update(doc.get('settings', {}))
    pages = [dict(uid=p['uid'], caption=p.get('caption', '')) for p in doc.get('pages', [])]
    return settings, pages, items


def add_layer_to_group(lyr, group_name, project=None):
    """Add a layer to the project inside a layer-tree group (created on top if needed), so
    every saved profile / model is its own layer under one group that is easy to remove."""
    from qgis.core import QgsProject
    proj = project or QgsProject.instance()
    proj.addMapLayer(lyr, False)
    root = proj.layerTreeRoot()
    grp = root.findGroup(group_name) or root.insertGroup(0, group_name)
    grp.insertLayer(0, lyr)
    return grp


def unique_layer_name(name, kind_prop_value, project=None):
    """name, or 'name (2)' ... if a saved layer of this kind already has it."""
    from qgis.core import QgsProject
    proj = project or QgsProject.instance()
    taken = {l.name() for l in proj.mapLayers().values()
             if l.customProperty("temviewer/kind") == kind_prop_value}
    if name not in taken:
        return name
    k = 2
    while f"{name} ({k})" in taken:
        k += 1
    return f"{name} ({k})"


def tv_gpkg_path(project=None):
    """GeoPackage that holds the saved TEM-Viewer profiles / models of a project:
    '<project name>_TEMViewer.gpkg' next to the project file, or - while the project has
    not been saved yet - a shared file in the QGIS profile folder."""
    from qgis.core import QgsProject, QgsApplication
    proj = project or QgsProject.instance()
    fn = proj.fileName()
    if fn and not fn.startswith('geopackage:') and os.path.isdir(os.path.dirname(fn) or '.'):
        base = os.path.splitext(os.path.basename(fn))[0] or "project"
        return os.path.join(os.path.dirname(fn), f"{base}_TEMViewer.gpkg")
    d = os.path.join(QgsApplication.qgisSettingsDirPath(), 'TEM-Viewer')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, 'TEMViewer_saved_layers.gpkg')


def _gpkg_table_names(path):
    from qgis.core import QgsVectorLayer
    if not os.path.isfile(path):
        return set()
    probe = QgsVectorLayer(path, "probe", "ogr")
    if not probe.isValid():
        return set()
    return {p.split('!!::!!')[1].lower() for p in probe.dataProvider().subLayers() if '!!::!!' in p}


def persist_layer(mem_lyr, group_name, kind_prop_value, project=None):
    """Write a (memory) layer into the project's TEM-Viewer GeoPackage and add the file-backed
    copy to the layer group. Memory layers lose their features when the project is closed,
    so saved profiles / models must live in a file to survive reopening the project.
    Returns (layer, path). Raises RuntimeError if the file cannot be written."""
    from qgis.core import QgsProject, QgsVectorLayer, QgsVectorFileWriter
    proj = project or QgsProject.instance()
    path = tv_gpkg_path(proj)
    taken = _gpkg_table_names(path)
    base = re.sub(r'[^0-9A-Za-z_]+', '_', mem_lyr.name()).strip('_')[:50] or "layer"
    base = f"{kind_prop_value}_{base}"
    table, k = base, 2
    while table.lower() in taken:
        table = f"{base}_{k}"
        k += 1

    opt = QgsVectorFileWriter.SaveVectorOptions()
    opt.driverName = 'GPKG'
    opt.layerName = table
    e = getattr(QgsVectorFileWriter, 'ActionOnExistingFile', None)
    act = 'CreateOrOverwriteLayer' if os.path.isfile(path) else 'CreateOrOverwriteFile'
    opt.actionOnExistingFile = getattr(e, act) if e is not None and hasattr(e, act) \
        else getattr(QgsVectorFileWriter, act)
    res = QgsVectorFileWriter.writeAsVectorFormatV3(mem_lyr, path, proj.transformContext(), opt)
    code = getattr(res[0], 'value', res[0])
    if code != 0:
        raise RuntimeError(f"Could not write '{table}' to {path}: {res[1]}")

    lyr = QgsVectorLayer(f"{path}|layername={table}", mem_lyr.name(), "ogr")
    if not lyr.isValid() or lyr.featureCount() < mem_lyr.featureCount():
        raise RuntimeError(f"The saved layer '{table}' in {path} could not be reopened.")
    lyr.setCustomProperty("temviewer/kind", kind_prop_value)
    add_layer_to_group(lyr, group_name, proj)
    return lyr, path


def count_layers_of_kind(kind_prop_value, project=None):
    from qgis.core import QgsProject
    proj = project or QgsProject.instance()
    return sum(1 for l in proj.mapLayers().values() if l.customProperty("temviewer/kind") == kind_prop_value)
