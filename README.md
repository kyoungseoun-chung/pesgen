# Python Every Street Generator

Python code to create paths to be used for #everystreet challenge. The code is inspired by [everystreet](https://github.com/matejker/everystreet).

## Disclaimer

Original code from `everystreet` has dependencies of author's own network library and `osmnx`. Since we only use a small portion of this library, recreate its functionality and try to make it as slim as possible.
    - Also, `osmnx` comes with `fiona` which requires external `GDAL` library (outside `pip` world).

## Dependencies

- `python = "^3.10"`
- `networkx = "^2.8.8"`: for graph manipulation
- `shapely = "^1.8.5"`: for polygon manipulation
- `pyproj = "^3.4.0"`: for map projection and back projection

- `numpy = "^1.23.5"`
- `matplotlib = "^3.6.2"`
