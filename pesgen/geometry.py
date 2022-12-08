#!/usr/bin/env python3
"""Geometry module contains Polygon. Adopt `shapely` library and clean up unnecessary parts."""
from __future__ import annotations

from copy import deepcopy
from math import floor
from typing import TypedDict

import pyproj
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

CRS_WGS84 = pyproj.CRS("EPSG:4326")
CRS_UTM = pyproj.CRS("EPSG:32618")


class GeoJSON(TypedDict):

    type: str
    """Geometry type. Should be polygon."""
    coordinates: list[list[list[float]]]
    """Coordinates extracted from open street map."""


class Polygon:
    """polygon geometry class to manipulate `shapely.geometry.Polygon`."""

    def __init__(self, geojson: GeoJSON):
        """Initialize Polygon class.

        Args:
            geojson (dict): GeoJSON of the polygon. It includes types and coordinates.
        """

        self.crs = CRS_WGS84

        self.geojson: GeoJSON = geojson
        # Somehow, shapely buffer returns BaseGeometry instead of Polygon. Therefore, additional type `BaseGeometry` is added.
        self._geo: ShapelyPolygon | BaseGeometry = ShapelyPolygon(
            self.geojson["coordinates"][0].copy()
        )

    def __repr__(self) -> str:

        return f"{self.__class__.__name__}(bounds={self.geometry.bounds}, area={self.geometry.area})"

    @property
    def geometry(self) -> ShapelyPolygon | BaseGeometry:
        """Return `shapely.geometry.Polygon`"""
        return self._geo

    @geometry.setter
    def geometry(self, other: ShapelyPolygon | BaseGeometry) -> None:
        """Set geometry from other."""
        self._geo = other

    @property
    def utm_zone(self) -> int:
        """Calculate UTM zone from avg longitude to define CRS to project to"""
        avg_lng = self.geometry.representative_point().x  # type: ignore

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
        for coord in self.geojson["coordinates"][0].copy():
            # round floating point lats and longs to 6 decimals (ie, ~100 mm)
            # so we can hash and cache strings consistently
            s = f"{s}{separator}{coord[1]:.6f}{separator}{coord[0]:.6f}"
        return s

    @property
    def swap_coord(self) -> list[list[float]]:
        latlng = []
        for coord in self.geojson["coordinates"][0].copy():
            latlng.append([coord[1], coord[0]])
        return latlng

    # TODO: there is a projection coordinate order issue.
    def to_utm(self) -> Polygon:
        """Convert latitude and longitude to UTM coordinates.

        Note:
            - `pyproj` uses lat-lng order for `CRS_UTM`. Therefore, we need to swap the coordinate. (On the other hand, openstreet map returns coordinates in lng-lat order.)
        """

        if self.crs == CRS_UTM:
            return self
        else:
            project = pyproj.Transformer.from_crs(
                CRS_WGS84, CRS_UTM, always_xy=True
            ).transform

            geo = transform(project, ShapelyPolygon(self.swap_coord))
            self.geojson["coordinates"][0] = [
                [coord[1], coord[0]] for coord in geo.exterior.coords
            ]
            # Reconstruction of polygon using swapped coordinates.
            self._geo = ShapelyPolygon(self.geojson["coordinates"][0].copy())
            self.crs = CRS_UTM
            return self

    def to_lnglat(self) -> Polygon:
        """Convert UTM coordinates to latitude and longitude."""

        if self.crs == CRS_WGS84:
            return self
        else:
            project = pyproj.Transformer.from_crs(
                CRS_UTM, CRS_WGS84, always_xy=True
            ).transform
            geo = transform(project, ShapelyPolygon(self.swap_coord))

            # Bring back to long-lat order
            self.geojson["coordinates"][0] = [
                [coord[1], coord[0]] for coord in geo.exterior.coords
            ]
            self._geo = ShapelyPolygon(self.geojson["coordinates"][0].copy())

            self.crs = CRS_WGS84

            return self

    def buffered(self, buffer: float) -> Polygon:
        """Buffer the polygon by a given distance.

        Args:
            buffer (float): Distance to buffer the polygon by.
        """

        self._geo = self._geo.buffer(buffer)
        self.geojson["coordinates"][0] = [
            [coord[0], coord[1]] for coord in self._geo.exterior.coords  # type: ignore
        ]

        return self

    def copy(self) -> Polygon:
        return deepcopy(self)
