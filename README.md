# TEM-Viewer

A QGIS plugin to view TEM / AEM 1D inversion models: sounding maps, resistivity (and IP) profiles, mean-resistivity maps and geological sections.

Works with QGIS 3.30 or newer (Qt5) and QGIS 4 (Qt6).

## Install

- **From the QGIS plugin repository** (once published): *Plugins → Manage and Install Plugins…*, search for **TEM-Viewer**.
- **From a zip**: download `tem_viewer.zip` from the Releases page, then *Plugins → Manage and Install Plugins… → Install from ZIP*.

The plugin appears under *Plugins → TEM-Viewer* and as a toolbar button.

Optional Python packages: `scipy` (faster neighbour search and variogram fitting for the mean maps) and `openpyxl` (Excel files). Everything works without them.

## Tabs

| Tab | What it does |
|---|---|
| **TEM data** | Import `.xyz` model files (plain column files, or files with a `/`-commented header; the coordinate system is read from the header when present) as sounding layers. Lists the sounding layers and colours soundings on the map by any attribute, with a colour bar on the map. |
| **Profile** | Sections along a line drawn on the map, through soundings selected on the map, along a line from a line layer, or along a saved profile. Bars or interpolated, DOI mask, data misfit, direction marks and a cursor linked to the map, colour scales per parameter, optional topography from a DEM raster and boreholes from a borehole point layer. Profiles can be saved to a GeoPackage and loaded again for editing. Export as PNG / PDF / SVG, A4 PDF with a location map, CSV or Excel. |
| **Mean maps** | Mean resistivity (or IP parameter) in depth or elevation slices, inverse-distance or ordinary-kriging interpolation, added to the map as GeoTIFF layers. |
| **Geological Model** | Pick the surfaces between formations on a section, name them and plot the geological section; export the surfaces as 3D lines. |

## Input files

Sounding files are column files with one row per sounding. Required: X / Y coordinates (`X`, `UTMX`, `EASTING`, …) and resistivities `RHO_1 … RHO_N` with either layer tops (`DEP_TOP_i`) or thicknesses (`THK_i`). Optional: `ELEVATION`, `DOI_STANDARD` / `DOI_CONSERVATIVE`, data misfit (`RESDATA`, `RESTOTAL`).

**IP parameters** are recognised in any letter case:

| Shown as | Column names |
|---|---|
| PHI (max. phase) | `PHI_i`, `PHIMAX_i`, `PHI_MAX_i`, `MPA_i` |
| TPHI (τφ) | `TPHI_i`, `TAU_PHI_i`, `TAUPHI_i` |
| C | `C_i`, `CEXP_i` |
| M0 (chargeability) | `M0_i`, `M_i`, `CHG_i`, `CHARGEABILITY_i` |
| τ | `TAU_i` |

Any other model parameter with one column per layer (`XYZ_1 … XYZ_N`) can be plotted as well.

## Building the plugin zip

```bash
zip -r tem_viewer.zip tem_viewer -x "*/__pycache__/*"
```

## License

GPL-2.0-or-later (see `LICENSE`).
