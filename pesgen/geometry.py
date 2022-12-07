#!/usr/bin/env python3
"""Geometry module contains Polygon. Adopt `shapely` library and clean up unnecessary parts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from math import floor

import pyproj
from shapely.geometry import mapping
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

CRS_WGS84 = pyproj.CRS("EPSG:4326")
CRS_UTM = pyproj.CRS("EPSG:32618")


@dataclass
class Polygon(ShapelyPolygon):
    """Custom polygon geometry class inherits from `shapely.geometry.Polygon`."""

    geojson: dict[str, str | list[list[list[float]]]]
    """GeoJSON of the polygon. It includes types and coordinates."""

    def __post_init__(self):
        super().__init__(self.geojson["coordinates"][0])

    @property
    def utm_zone(self) -> int:
        """Calculate UTM zone from avg longitude to define CRS to project to"""
        avg_lng = self.representative_point().x  # type: ignore

        return int(floor((avg_lng + 180) / 6) + 1)

    @property
    def utm_crs(self) -> str:
        """Coordinate Reference System of the polygon in UTM"""
        return f"+proj=utm +zone={self.utm_zone} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

    @property
    def osm_coord(self) -> str:
        """Convert coordinates to OSM format. Partially copied from `osmnx` package."""
        s = ""
        separator = " "
        for coord in self.geojson["coordinates"][0]:
            # round floating point lats and longs to 6 decimals (ie, ~100 mm)
            # so we can hash and cache strings consistently
            s = f"{s}{separator}{coord[1]:.6f}{separator}{coord[0]:.6f}"
        return s

    def to_utm(self) -> Polygon:
        """Convert latitude and longitude to UTM coordinates."""
        project = pyproj.Transformer.from_crs(
            CRS_WGS84, CRS_UTM, always_xy=True
        ).transform
        shapely_object = transform(
            project, ShapelyPolygon(self.geojson["coordinates"][0])
        )

        return Polygon(_convert_to_geojson(shapely_object))

    def to_latlng(self) -> Polygon:
        """Convert UTM coordinates to latitude and longitude."""
        project = pyproj.Transformer.from_crs(
            CRS_UTM, CRS_WGS84, always_xy=True
        ).transform
        shapely_object = transform(
            project, ShapelyPolygon(self.geojson["coordinates"][0])
        )
        return Polygon(_convert_to_geojson(shapely_object))

    def buffered(self, buffer: float) -> Polygon:
        """Buffer the polygon by a given distance.

        Args:
            buffer (float): Distance to buffer the polygon by.

        Returns:
            Polygon: Buffered polygon.
        """

        shapely_object = ShapelyPolygon(self.geojson["coordinates"][0]).buffer(
            buffer
        )

        return Polygon(_convert_to_geojson(shapely_object))


def _convert_to_geojson(obj: ShapelyPolygon | BaseGeometry):
    """Convert shapely object to GeoJSON.

    Args:
        obj (ShapelyPolygon): Shapely object to convert to GeoJSON.

    Returns:
        dict[str, str | list[list[list[float]]]]: GeoJSON of the polygon.
    """

    return json.loads(json.dumps(mapping(obj)))
