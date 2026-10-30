# -*- coding: utf-8 -*-
"""TEM-Viewer - view TEM/AEM 1D inversion models in QGIS."""


def classFactory(iface):
    from .tem_viewer import TemViewerPlugin
    return TemViewerPlugin(iface)
