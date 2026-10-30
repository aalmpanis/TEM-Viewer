# -*- coding: utf-8 -*-
"""
TEM-Viewer - boreholes drawn in a section (Profile tab, Geological Model tab).

Boreholes within a given distance of a profile line are projected onto it and
drawn as narrow lithology columns on top of the resistivity section.
(Shared by the main window and the stand-alone Profile Tool.)
"""
import zlib

import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.collections
import matplotlib.transforms

from qgis.core import QgsProject, QgsCoordinateTransform, QgsPointXY

# ---------------------------------------------------------------------------
#  Lithology colours by grain size / hardness
#
#  Every description is placed on one "coarseness / hardness" index:
#      0 clay · 1 silt · 1.6 till / mixed · 2.2 fine sand · 3 sand · 4 gravel
#      5 weathered / soft rock · 6 hard bedrock
#  and coloured along a ramp that follows the usual resistivity colours
#  (fine, conductive = blue → sand = yellow/red → bedrock = purple).
#  The main (head) noun counts most, modifiers ("sandy", "with gravel") shift it.
#  Organic soils and fill/topsoil get their own colours.
# ---------------------------------------------------------------------------
GRAIN_INDEX = {
    'clay': 0.0, 'mud': 0.1, 'gyttja': 0.2, 'marl': 0.5, 'loam': 0.9, 'silt': 1.0, 'loess': 1.0,
    'till': 1.6, 'diamict': 1.6, 'diamicton': 1.6, 'moraine': 1.6, 'boulder clay': 1.3,
    'fine sand': 2.2, 'sand': 3.0, 'medium sand': 3.0, 'coarse sand': 3.5,
    'gravel': 4.0, 'pebble': 4.1, 'pebbles': 4.1, 'cobble': 4.3, 'cobbles': 4.3, 'boulder': 4.5,
    'boulders': 4.5, 'stone': 4.3, 'stones': 4.3,
    # weathered / soft rock
    'saprolite': 5.0, 'laterite': 4.8, 'regolith': 4.9, 'weathered rock': 5.0, 'ash': 5.1,
    'pyroclastic': 5.3, 'pyroclastics': 5.3, 'tuff': 5.3, 'pumice': 5.3, 'scoria': 5.4,
    'lapilli': 5.3, 'tephra': 5.2, 'pyroclast': 5.3, 'ler': 0.0, 'grus': 4.0,
    'shale': 5.2, 'mudstone': 5.2, 'claystone': 5.2, 'siltstone': 5.4, 'chalk': 5.4,
    'agglomerate': 5.6, 'breccia': 5.7, 'sandstone': 5.7, 'conglomerate': 5.8, 'volcanic': 5.8,
    # hard bedrock
    'limestone': 5.85, 'dolomite': 5.9, 'schist': 5.9, 'ignimbrite': 5.9, 'trachyte': 6.0,
    'phonolite': 6.0, 'rhyolite': 6.0, 'andesite': 6.0, 'basalt': 6.0, 'lava': 6.0, 'dolerite': 6.0,
    'diabase': 6.0, 'gabbro': 6.0, 'granite': 6.0, 'gneiss': 6.0, 'quartzite': 6.0, 'rock': 6.0,
    'bedrock': 6.0, 'basement': 6.0, 'crystalline': 6.0,
}
SPECIAL = {
    'peat': (95, 70, 40), 'organic': (95, 70, 40), 'humus': (95, 70, 40), 'lignite': (60, 45, 30),
    'coal': (45, 40, 35), 'topsoil': (150, 135, 110), 'soil': (150, 135, 110), 'fill': (140, 140, 140),
    'made ground': (140, 140, 140), 'anthropogenic': (140, 140, 140), 'water': (200, 225, 245),
    'void': (235, 235, 235), 'cavity': (235, 235, 235),
}
ADJECTIVES = {'clayey': 'clay', 'silty': 'silt', 'sandy': 'sand', 'gravelly': 'gravel', 'gravely': 'gravel',
              'stony': 'stone', 'pebbly': 'pebble', 'bouldery': 'boulder', 'loamy': 'loam',
              'peaty': 'peat', 'marly': 'marl', 'tuffaceous': 'tuff', 'chalky': 'chalk',
              'basaltic': 'basalt', 'pumiceous': 'pumice', 'ignimbritic': 'ignimbrite'}
INDEX_RAMP = [(0.0, (40, 70, 200)), (1.0, (60, 160, 220)), (1.6, (80, 190, 150)), (2.2, (185, 215, 80)),
              (2.7, (250, 220, 60)), (3.2, (245, 150, 40)), (3.8, (225, 60, 40)), (4.5, (160, 30, 30)),
              (5.0, (215, 105, 170)), (5.5, (180, 90, 200)), (6.0, (110, 50, 160))]
_MULTI = sorted([k for k in list(GRAIN_INDEX) + list(SPECIAL) if ' ' in k], key=len, reverse=True)
_SINGLE = sorted([k for k in list(GRAIN_INDEX) + list(SPECIAL) if ' ' not in k])


def _ramp(idx):
    idx = min(max(idx, 0.0), 6.0)
    for (a, ca), (b, cb) in zip(INDEX_RAMP[:-1], INDEX_RAMP[1:]):
        if idx <= b:
            t = (idx - a) / (b - a) if b > a else 0.0
            return tuple(int(round(ca[i] + t * (cb[i] - ca[i]))) for i in range(3))
    return INDEX_RAMP[-1][1]


def _word(tok):
    """Known lithology word for a token (exact, adjective, plural or a clear abbreviation)."""
    if tok in GRAIN_INDEX or tok in SPECIAL:
        return tok, False
    if tok in ADJECTIVES:
        return ADJECTIVES[tok], True
    if tok.endswith('s') and (tok[:-1] in GRAIN_INDEX or tok[:-1] in SPECIAL):
        return tok[:-1], False
    for suf in ('ler', 'sand', 'grus', 'silt'):        # Danish compounds: moræneler, smeltevandssand
        if len(tok) >= len(suf) + 3 and tok.endswith(suf):
            return suf, False
    if len(tok) >= 3:                                  # e.g. 'cla' -> clay, 'grav' -> gravel
        hits = sorted([k for k in _SINGLE if k.startswith(tok)], key=len)
        if hits and all(h.startswith(hits[0]) for h in hits):   # 'cla' -> clay (not claystone)
            return hits[0], False
    return None, False


def _parse(name):
    """Lithology words of a description, in order: [(position, word, is_admixture)]."""
    text = ' ' + ''.join(c if c.isalpha() else ' ' for c in str(name or '').lower()) + ' '
    found = []                                         # (position, word, is_adjective, joiner)
    for m in _MULTI:
        pos = text.find(' ' + m + ' ')
        if pos >= 0:
            found.append((pos, m, False))
            text = text[:pos] + ' ' + ' ' * len(m) + text[pos + len(m) + 1:]
    pos = 0
    minor = False                     # after "with", "some", "traces of" ... words are admixtures
    for tok in text.split(' '):
        if tok:
            if tok in ('with', 'some', 'minor', 'trace', 'traces', 'occasional', 'few'):
                minor = True
            elif tok in ('and', 'or', 'of'):
                pass
            elif tok == 'weathered':
                found.append((pos, 'weathered', True))
            else:
                w, adj = _word(tok)
                if w:
                    found.append((pos, w, adj or minor))
        pos += len(tok) + 1
    found.sort()
    return found


def head_word(name):
    """Main lithology noun of a description ('sandy clay' -> 'clay'), None if unknown."""
    found = _parse(name)
    if not found:
        return None
    nouns = [w for _, w, adj in found if not adj and w != 'weathered']
    return nouns[0] if nouns else found[0][1]


def classify_lithology(name):
    """('grain', index) / ('special', rgb) / (None, None) for a lithology description."""
    found = _parse(name)
    if not found:
        return None, None
    nouns = [w for _, w, adj in found if not adj and w != 'weathered']
    head = nouns[0] if nouns else found[0][1]
    if head in SPECIAL:
        return 'special', SPECIAL[head]
    weathered = any(w == 'weathered' for _, w, _ in found)
    if head == 'weathered':
        return 'grain', 5.0
    total, wsum = 0.0, 0.0
    for _, w, adj in found:
        if w in SPECIAL or w == 'weathered':
            continue
        weight = 1.0 if w == head else (0.3 if adj else 0.6)
        total += weight * GRAIN_INDEX[w]
        wsum += weight
    idx = total / wsum if wsum else GRAIN_INDEX.get(head, 3.0)
    if GRAIN_INDEX.get(head, 0) >= 4.8:                # rock head: stays rock
        idx = max(idx, 4.8)
    if weathered and idx >= 5.0:
        idx = 5.0
    return 'grain', idx


# ---------------------------------------------------------------------------
#  Lithology colours: fixed group colours + user choices
#
#  Every description gets its colour from its main noun ("sandy clay" -> clay):
#      clay blue · sand orange · gravel red · basalt purple · tuff grey ·
#      ignimbrite pink · pyroclastic light brown
#  The plain name gets the group colour itself; other descriptions of the same group
#  ("sandy clay", "clay with gravel") get a lighter / darker shade of it, picked from the
#  text, so a description always has the same colour – in every tab, project and session.
#  Other lithologies keep the grain-size ramp above. Colours chosen by the user (Boreholes
#  tab > Lithology colours) are stored in the QGIS settings and win over all of this.
# ---------------------------------------------------------------------------
LITH_GROUPS = [            # (group, base rgb, nouns that belong to it)
    ('clay', (31, 95, 200), ('clay', 'ler', 'mud', 'claystone', 'boulder clay')),
    ('sand', (245, 140, 30), ('sand', 'fine sand', 'medium sand', 'coarse sand')),
    ('gravel', (210, 35, 35), ('gravel', 'grus', 'pebble', 'cobble', 'boulder', 'stone')),
    ('basalt', (120, 55, 170), ('basalt', 'lava', 'dolerite', 'diabase')),
    ('tuff', (140, 140, 140), ('tuff',)),
    ('ignimbrite', (232, 120, 190), ('ignimbrite',)),
    ('pyroclastic', (200, 162, 118), ('pyroclastic', 'pyroclastics', 'pyroclast', 'pumice', 'scoria',
                                      'lapilli', 'tephra', 'ash', 'agglomerate')),
]
GROUP_OF = {w: g for g, _, words in LITH_GROUPS for w in words}
GROUP_RGB = {g: rgb for g, rgb, _ in LITH_GROUPS}
SETTINGS_KEY = "TEM-Viewer/lithology_colors"
_SHADES = [(14, 0), (-14, 0), (24, 6), (-22, -6), (8, 10), (-8, -10), (30, -4), (-28, 4),
           (18, -12), (-18, 12)]          # (lightness, hue degrees) steps for variants
_CACHE = {}
_OVERRIDES = None
_EVENTS = None
_WHITE = (0.95047, 1.0, 1.08883)


def _rgb_to_lab(rgb):
    c = np.array(rgb, float) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = (m @ c) / np.array(_WHITE)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def _lab_to_rgb(lab):
    L, a, b = lab
    fy = (L + 16) / 116
    f = np.array([fy + a / 500, fy, fy - b / 200])
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * np.array(_WHITE)
    m = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    c = m @ xyz
    c = np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.sign(c) * np.abs(c) ** (1 / 2.4) - 0.055)
    rgb = np.clip(c, 0, 1)
    inside = bool(np.all((c >= -0.002) & (c <= 1.002)))
    return tuple(int(round(v * 255)) for v in rgb), inside


def _key(name):
    return ' '.join(str(name or '').lower().split())


def _hex(rgb):
    return '#%02x%02x%02x' % tuple(int(v) for v in rgb)


def _from_hex(h):
    h = str(h).lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# ---- user colours (QGIS settings, shared by all projects) ------------------
def _load_overrides():
    global _OVERRIDES
    if _OVERRIDES is None:
        _OVERRIDES = {}
        try:
            import json
            from qgis.core import QgsSettings
            raw = QgsSettings().value(SETTINGS_KEY, "")
            if raw:
                _OVERRIDES = {k: _from_hex(v) for k, v in json.loads(str(raw)).items()}
        except Exception:
            _OVERRIDES = {}
    return _OVERRIDES


def _save_overrides():
    try:
        import json
        from qgis.core import QgsSettings
        QgsSettings().setValue(SETTINGS_KEY, json.dumps({k: _hex(v) for k, v in _load_overrides().items()}))
    except Exception:
        pass


def color_events():
    """QObject whose `changed` signal fires when the user changes a lithology colour."""
    global _EVENTS
    if _EVENTS is None:
        from qgis.PyQt.QtCore import QObject, pyqtSignal

        class _Events(QObject):
            changed = pyqtSignal()
        _EVENTS = _Events()
    return _EVENTS


def _notify():
    _CACHE.clear()
    _save_overrides()
    try:
        color_events().changed.emit()
    except Exception:
        pass


def set_lith_color(name, rgb):
    """User colour for one description (rgb None = back to the default)."""
    k = _key(name)
    if rgb is None:
        _load_overrides().pop(k, None)
    else:
        _load_overrides()[k] = tuple(int(v) for v in rgb)
    _notify()


def set_group_color(group, rgb):
    """User colour for a whole group (all its descriptions follow, unless set one by one)."""
    if rgb is None:
        _load_overrides().pop('@' + group, None)
    else:
        _load_overrides()['@' + group] = tuple(int(v) for v in rgb)
    _notify()


def reset_lith_colors():
    _load_overrides().clear()
    _notify()


def user_colored(name):
    return _key(name) in _load_overrides()


def group_color(group):
    return _load_overrides().get('@' + group, GROUP_RGB.get(group))


def lith_group(name):
    """Colour group of a description ('sandy clay' -> 'clay'), None if it has none."""
    k = _key(name)
    if not k:
        return None
    head = head_word(k)
    if head in GROUP_OF:
        return GROUP_OF[head]
    for _, w, adj in _parse(k):                    # e.g. 'welded tuff' -> tuff
        if not adj and w in GROUP_OF:
            return GROUP_OF[w]
    return None


def _shade(base, k):
    """A variant of base for description k: the same every time (from the text)."""
    L, a, b = _rgb_to_lab(base)
    C, H = float(np.hypot(a, b)), float(np.arctan2(b, a))
    i = zlib.crc32(k.encode('utf-8')) % len(_SHADES)
    for j in range(len(_SHADES)):
        dL, dH = _SHADES[(i + j) % len(_SHADES)]
        h2 = H + dH * np.pi / 180.0
        rgb, ok = _lab_to_rgb((min(max(L + dL, 20), 92), C * np.cos(h2), C * np.sin(h2)))
        if ok:
            return rgb
    return base


def _word_rgb(w):
    """Colour of a single lithology word (for the admixtures of a description)."""
    if w in GROUP_OF:
        return group_color(GROUP_OF[w])
    if w in SPECIAL:
        return SPECIAL[w]
    if w in GRAIN_INDEX:
        return _ramp(GRAIN_INDEX[w])
    return None


def _default_rgb(k):
    grp = lith_group(k)
    if grp is not None:
        base = np.array(group_color(grp), float)
        found = _parse(k)
        members = {w for w, g in GROUP_OF.items() if g == grp}
        head_seen, mix, wsum = False, np.zeros(3), 0.0
        for _, w, adj in found:               # admixtures pull the colour a little towards theirs
            if w in members and not head_seen:
                head_seen = True
                continue
            c = _word_rgb(w)
            if c is None or (w in members):
                continue
            wt = 0.15 if adj else 0.25
            mix += wt * np.array(c, float)
            wsum += wt
        if wsum > 0:
            wsum = min(wsum, 0.4)
            rgb = tuple(int(round(v)) for v in (1 - wsum) * base + mix / max(sum(
                [0.15 if a_ else 0.25 for _, w_, a_ in found if _word_rgb(w_) is not None and w_ not in members]
            ), 1e-9) * wsum)
            if np.linalg.norm(_rgb_to_lab(rgb) - _rgb_to_lab(base)) < 10:
                rgb = _shade(rgb, k)            # keep variants visibly apart from the plain name
            return rgb
        single = {m for m in members if ' ' not in m}
        toks = ''.join(c if c.isalpha() else ' ' for c in k).split()
        plain = bool(toks) and all(t in single or (t.endswith('s') and t[:-1] in single) for t in toks)
        return tuple(int(v) for v in base) if plain else _shade(tuple(int(v) for v in base), k)
    kind, v = classify_lithology(k)
    if kind == 'special':
        return v
    if kind == 'grain':
        return _ramp(v)
    return None


def lith_rgb_or_none(name):
    """Colour of a description, None when it is not a known lithology (and not set by the user)."""
    k = _key(name)
    if not k:
        return None
    ov = _load_overrides().get(k)
    if ov is not None:
        return tuple(ov)
    return _default_rgb(k)


def register_lithologies(names):
    """Kept for older callers: colours no longer depend on which other names are in use."""
    for n in names:
        lith_rgb(n)


def lith_rgb(name):
    """Colour (0-255 RGB) of a lithology description – always the same for the same text."""
    k = _key(name)
    if k in _CACHE:
        return _CACHE[k]
    rgb = lith_rgb_or_none(k)
    if rgb is None:                                  # unknown: a pastel of a name-dependent hue
        h = zlib.crc32(k.encode('utf-8'))
        ang = (h % 360) * np.pi / 180.0
        rgb = _lab_to_rgb(np.array([78.0, 32 * np.cos(ang), 32 * np.sin(ang)]))[0] if k else (200, 200, 200)
    _CACHE[k] = tuple(int(v) for v in rgb)
    return _CACHE[k]


def project_on_line(xy, line, extend=False):
    """Distance along the polyline and perpendicular offset of points xy (n x 2).
    extend=True: points beyond the ends are projected onto the prolongation of the
    first / last segment, so a point 50 m past the end gets distance L + 50 (or -50
    before the start) and its true perpendicular offset."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    line = np.asarray(line, dtype=float)
    a, b = line[:-1], line[1:]
    seg = b - a
    L = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(L)])
    best_d = np.full(len(xy), np.inf)
    best_s = np.zeros(len(xy))
    for k in range(len(seg)):
        if L[k] == 0:
            continue
        t = ((xy - a[k]) @ seg[k]) / L[k] ** 2
        lo = -np.inf if (extend and k == 0) else 0.0
        hi = np.inf if (extend and k == len(seg) - 1) else 1.0
        t = np.clip(t, lo, hi)
        c = a[k] + t[:, None] * seg[k]
        d = np.hypot(xy[:, 0] - c[:, 0], xy[:, 1] - c[:, 1])
        better = d < best_d
        best_d[better] = d[better]
        best_s[better] = cum[k] + t[better] * L[k]
    return best_s, best_d


def boreholes_near_line(registry, layers, line, crs, max_dist):
    """Boreholes of `layers` within max_dist (m) of the polyline `line` (n x 2, in crs).
    Returns dicts sorted along the line: name, dist, offset, elev, depth, intervals."""
    out = []
    line = np.asarray(line, dtype=float)
    for lyr in layers:
        try:
            bh, logs = registry.boreholes_data(lyr)
        except Exception:
            continue
        if bh is None or bh.empty:
            continue
        x = pd.to_numeric(bh['XUTM'], errors='coerce').to_numpy(dtype=float)
        y = pd.to_numeric(bh['YUTM'], errors='coerce').to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if crs is not None and crs.isValid() and lyr.crs().isValid() and lyr.crs() != crs:
            tr = QgsCoordinateTransform(lyr.crs(), crs, QgsProject.instance())
            for i in np.where(ok)[0]:
                p = tr.transform(QgsPointXY(float(x[i]), float(y[i])))
                x[i], y[i] = p.x(), p.y()
        if not ok.any():
            continue
        pts = np.column_stack([x[ok], y[ok]])
        s, _off_ext = project_on_line(pts, line, extend=True)     # position along the line (may be beyond an end)
        _, off = project_on_line(pts, line, extend=False)         # true distance to the line itself
        idx = np.where(ok)[0]
        total = float(np.sum(np.hypot(*np.diff(line, axis=0).T)))
        logs = logs if logs is not None else pd.DataFrame(columns=['BoreholeNo'])
        # true distance to the line (to its end point beyond the ends) within max_dist
        for k in np.where(off <= max_dist)[0]:
            r = bh.iloc[idx[k]]
            name = str(r.get('BoreholeNo'))
            sub = logs[logs['BoreholeNo'].astype(str) == name] if not logs.empty else logs
            intervals = []
            for _, lr in sub.sort_values('Top').iterrows() if not sub.empty else []:
                t, b = pd.to_numeric(lr.get('Top'), errors='coerce'), pd.to_numeric(lr.get('Bottom'), errors='coerce')
                if pd.isna(t) or pd.isna(b) or b <= t:
                    continue
                lith = lr.get('RockType')
                if lith is None or (isinstance(lith, float) and np.isnan(lith)) or str(lith).strip() in ('', 'nan'):
                    lith = lr.get('LayerDescription')
                intervals.append((float(t), float(b), str(lith or '').strip()))
            depth = pd.to_numeric(r.get('DrillDepth'), errors='coerce')
            if intervals:
                depth = max(depth if pd.notna(depth) else 0.0, intervals[-1][1])
            elev = pd.to_numeric(r.get('Elevation'), errors='coerce')
            out.append(dict(name=name, dist=float(s[k]), offset=float(off[k]),
                            elev=float(elev) if pd.notna(elev) else np.nan,
                            depth=float(depth) if pd.notna(depth) else np.nan,
                            intervals=intervals, layer=lyr.name(), layer_id=lyr.id(),
                            x=float(x[idx[k]]), y=float(y[idx[k]])))
    out.sort(key=lambda d: d['dist'])
    return out


def draw_boreholes(ax, bhs, width, to_y, labels=True, zorder=8):
    """Draw boreholes as lithology columns. to_y(bh, depth_array) -> y values.
    Returns {lithology: colour} of what was drawn (for a legend)."""
    used = {}
    register_lithologies([lith for b in bhs for _, _, lith in b['intervals']])
    for b in bhs:
        x0 = b['dist'] - width / 2.0
        if b['intervals']:
            verts, cols = [], []
            for t, bot, lith in b['intervals']:
                y0, y1 = to_y(b, np.array([t, bot]))
                verts.append([(x0, y0), (x0 + width, y0), (x0 + width, y1), (x0, y1)])
                c = tuple(v / 255.0 for v in lith_rgb(lith))
                cols.append(c)
                used.setdefault(lith or '(no description)', c)
            ax.add_collection(matplotlib.collections.PolyCollection(
                verts, facecolors=cols, edgecolors='black', linewidths=0.4, zorder=zorder))
            y_top, y_bot = to_y(b, np.array([0.0, b['intervals'][-1][1]]))
        else:
            # no log (yet): an empty dashed column – to the drill depth if known, else a short stub
            depth = b['depth'] if np.isfinite(b['depth']) and b['depth'] > 0 else EMPTY_STUB_M
            y_top, y_bot = to_y(b, np.array([0.0, depth]))
            ax.add_patch(mpatches.Rectangle((x0, min(y_top, y_bot)), width, abs(y_bot - y_top),
                                            facecolor='white', edgecolor='0.35', linewidth=0.7,
                                            linestyle='--', alpha=0.9, zorder=zorder))
        if labels:
            ax.annotate(b['name'], (b['dist'], max(y_top, y_bot) if y_top >= y_bot else min(y_top, y_bot)),
                        xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                        fontsize=7, rotation=90, zorder=zorder + 1, clip_on=True,
                        bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.7))
    return used


def draw_sounding_frames(ax, left, right, yt, yb, zorder=4):
    """Outline of every sounding column (and faint layer boundaries) on an
    interpolated section, so it is clear where each sounding is."""
    left, right = np.asarray(left, float), np.asarray(right, float)
    yt, yb = np.asarray(yt, float), np.asarray(yb, float)
    frames, layers = [], []
    for i in range(len(left)):
        ok = np.isfinite(yt[i]) & np.isfinite(yb[i])
        if not ok.any():
            continue
        t, b = yt[i][ok], yb[i][ok]
        top = float(np.max(np.maximum(t, b)))
        bot = float(np.min(np.minimum(t, b)))
        l_, r_ = left[i], right[i]
        frames.append([(l_, top), (r_, top), (r_, bot), (l_, bot), (l_, top)])
        for y in np.unique(np.concatenate([t, b])):
            if bot < y < top:
                layers.append([(l_, y), (r_, y)])
    if layers:
        ax.add_collection(matplotlib.collections.LineCollection(
            layers, colors='0.2', linewidths=0.35, alpha=0.3, zorder=zorder))
    if frames:
        ax.add_collection(matplotlib.collections.LineCollection(
            frames, colors='black', linewidths=0.6, alpha=0.45, zorder=zorder))


WL_COLOR = '#0057b8'
EMPTY_STUB_M = 20.0     # length of the empty column of a borehole with no log and no depth


def WL_HANDLE():
    """Legend entry for the water level."""
    from matplotlib.lines import Line2D
    return Line2D([], [], color=WL_COLOR, lw=1.8, marker='v', markersize=6, markerfacecolor='white',
                  markeredgecolor=WL_COLOR, label='Water level')


def draw_water_levels(ax, bhs, width, wl_y, zorder=10):
    """Water level in each borehole: a blue line across the column with the ▽ symbol
    above it. wl_y(bh) gives its y in the plot (nan = no water level). Returns the count."""
    segs, xs, ys = [], [], []
    for b in bhs:
        y = wl_y(b)
        if y is None or not np.isfinite(y):
            continue
        half = 0.75 * width
        segs.append([(b['dist'] - half, y), (b['dist'] + half, y)])
        xs.append(b['dist'])
        ys.append(y)
    if not segs:
        return 0
    ax.add_collection(matplotlib.collections.LineCollection(
        segs, colors='white', linewidths=3.2, zorder=zorder, capstyle='butt'))
    ax.add_collection(matplotlib.collections.LineCollection(
        segs, colors=WL_COLOR, linewidths=1.8, zorder=zorder + 0.1, capstyle='butt'))
    # ▽ just above the line (offset in points, so it looks the same at any zoom)
    for x, y in zip(xs, ys):
        ax.plot([x], [y], marker='v', markersize=7, markerfacecolor='white', markeredgecolor=WL_COLOR,
                markeredgewidth=1.4, linestyle='none', zorder=zorder + 0.2,
                transform=ax.transData + matplotlib.transforms.ScaledTranslation(
                    0, 6 / 72.0, ax.figure.dpi_scale_trans))
    return len(segs)


def lith_at(bhs, width, x, depth_of):
    """Lithology under the cursor: depth_of(bh) gives the depth at the cursor for that borehole."""
    for b in bhs:
        if abs(x - b['dist']) <= width / 2.0:
            d = depth_of(b)
            for t, bot, lith in b['intervals']:
                if t <= d < bot:
                    return f"{b['name']}: {lith} ({t:g}–{bot:g} m)"
            return b['name']
    return ""
