#!/usr/bin/env python3
"""Test downloader.py"""
from pesgen.downloader import osm_network
from pesgen.downloader import query_to_gd


def test_get_data_from_nominatim():
    location = "Grange, Edinburgh, Scotland"

    data = query_to_gd(location)
    polygon = data["geometry"]
    new_polygon = polygon.to_utm()
    buffered = new_polygon.buffered(500)
    new_new_polygon = new_polygon.to_latlng()

    network = osm_network(polygon)
    pass
