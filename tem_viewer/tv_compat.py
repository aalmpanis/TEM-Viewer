# -*- coding: utf-8 -*-
"""
TEM-Viewer - compatibility between QGIS 3 (Qt5 / PyQt5) and QGIS 4 (Qt6 / PyQt6).

Everything that differs between the two lives here, so the rest of the code can
use one spelling:
  * QAction                  (QtWidgets in Qt5, QtGui in Qt6)
  * field types for QgsField (QVariant.Type in Qt5, QMetaType.Type in Qt6)
  * matplotlib Qt backend    (qt5agg vs qtagg)
  * int flags for QPainter.drawText
  * QGIS enums whose scoped name differs between versions
"""
import functools

# --- QAction ---------------------------------------------------------------
try:
    from qgis.PyQt.QtGui import QAction          # Qt6
except ImportError:
    from qgis.PyQt.QtWidgets import QAction      # Qt5

# --- QgsField types --------------------------------------------------------
try:
    from qgis.PyQt.QtCore import QVariant
except ImportError:                              # pragma: no cover
    QVariant = None

if QVariant is not None and hasattr(QVariant, 'Double'):      # Qt5
    FIELD_DOUBLE, FIELD_INT, FIELD_STRING = QVariant.Double, QVariant.Int, QVariant.String
else:                                                         # Qt6 (QGIS 4)
    from qgis.PyQt.QtCore import QMetaType
    FIELD_DOUBLE = QMetaType.Type.Double
    FIELD_INT = QMetaType.Type.Int
    FIELD_STRING = QMetaType.Type.QString

# --- matplotlib Qt backend -------------------------------------------------
import matplotlib
try:
    matplotlib.use('QtAgg')
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
except Exception:                                # old matplotlib (< 3.5)
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT


def mpl_cmap(name):
    """Colormap by name for old (cm.get_cmap) and new (matplotlib.colormaps) matplotlib."""
    reg = getattr(matplotlib, 'colormaps', None)
    if reg is not None:
        return reg[name]
    from matplotlib import cm as _cm
    return _cm.get_cmap(name)


# --- flags as plain int (QPainter.drawText etc.) ---------------------------
def qflags(*flags):
    """Combine Qt flags of possibly different enum types into an int
    (PyQt6 enums are not ints and cannot be OR-ed across types)."""
    return functools.reduce(lambda a, f: a | int(getattr(f, 'value', f)), flags, 0)


# --- QGIS enums whose scoped spelling is not available everywhere -----------
def qgis_enum(cls, scoped, *names):
    """Return cls.<scoped>.<name> (QGIS 4) or the first existing cls.<name> (QGIS 3)."""
    group = getattr(cls, scoped, None)
    for n in names:
        if group is not None and hasattr(group, n):
            return getattr(group, n)
    for n in names:
        if hasattr(cls, n):
            return getattr(cls, n)
    raise AttributeError(f"{cls.__name__}: none of {names} found")
