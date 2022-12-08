#!/usr/bin/env python3
"""Download map data using Nominatim API."""
import json
import time
from collections import OrderedDict
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

from pesgen.geometry import Polygon

NOMINATIM_ENDPOINT = "https://nominatim.openstreetmap.org/"
OVERPASS_ENDPOINT = "https://overpass-api.de/api/"

# Filter used for Every Street Challenge
ES_FILTER = (
    '["highway"]["area"!~"yes"]["highway"!~"bridleway|bus_guideway|bus_stop|construction|'
    "cycleway|elevator|footway|motorway|motorway_junction|motorway_link|escalator|proposed|"
    'construction|platform|raceway|rest_area|path|service"]["access"!~"customers|no|private"]'
    '["public_transport"!~"platform"]["fee"!~"yes"]["foot"!~"no"]["service"!~"drive-through|'
    'driveway|parking_aisle"]["toll"!~"yes"]'
)


def query_to_gd(query: str):
    """Geocode a single query to geometric data to be proceed

    Args:
        query (str): Query to send to Nominatim API.
    """

    result = osm_place(query)

    geom_type = result["geojson"]["type"]
    if geom_type not in {"Polygon", "MultiPolygon"}:
        msg = f"OSM: Nominatim geocoder returned a {geom_type} as the geometry for query '{query}'"
        raise RuntimeError(msg)

    # build the GeoJSON feature from the chosen result
    south, north, west, east = result["boundingbox"]
    feature = {
        "type": "Feature",
        "geometry": Polygon(result["geojson"]),
        "properties": {
            "bbox_north": north,
            "bbox_south": south,
            "bbox_east": east,
            "bbox_west": west,
        },
    }
    # add the other attributes we retrieved
    for attr in result:
        if attr not in {
            "address",
            "boundingbox",
            "geojson",
            "icon",
            "licence",
        }:
            feature["properties"][attr] = result[attr]

    return feature


def osm_place(query: str) -> dict:
    """Request place data from Nominatim API.
    Code is copied and cleaned up from `osmnx` package. (see https://github.com/gboeing/osmnx)
        - More specifically, this function is a modified version of `osmnx.downloader._osm_place_download`

    Args:
        query (str): Query to send to Nominatim API.
    """

    params: OrderedDict[str, str | int] = OrderedDict()
    params["format"] = "json"
    params["polygon_geojson"] = 1
    # prevent OSM from deduping so we get precise number of results
    params["dedupe"] = 0
    # Max number of results to return
    params["limit"] = 1

    # Below is intended to be used in requests.
    # How to convert for the native urllib?
    if isinstance(query, str):
        params["q"] = query
    elif isinstance(query, dict):
        # add query keys in alphabetical order so URL is the same string
        # each time, for caching purposes
        for key in sorted(query):
            params[key] = query[key]
    else:  # pragma: no cover
        raise TypeError("query must be a dict or a string")

    query_string = urlencode(params)

    url = NOMINATIM_ENDPOINT + "search?" + query_string

    return _data_from_url(url, query)[0]


def osm_network(polygon: Polygon) -> dict:
    """Request place data from Overpass API.
    Code is copied and cleaned up from `osmnx` package. (see https://github.com/gboeing/osmnx)
        - More specifically, this function is a modified version of `osmnx.downloader._osm_network_download`

    Args:
        polygon (Polygon): `Polygon` object to send to Overpass API.
    """

    params: dict = {}
    params[
        "data"
    ] = f"[out:json][timeout:180];(way{ES_FILTER}(poly:'{polygon.osm_coord}');>;);out;"

    query_string = urlencode(params)

    url = OVERPASS_ENDPOINT + "interpreter?" + query_string

    return _data_from_url(url, polygon)


def _data_from_url(url: str, query: str | Polygon) -> dict:
    """Retrive data from url."""

    sc = 400

    retrieved_data: dict = {}

    while sc != 200:

        try:
            response = urlopen(url)
            sc = response.status
            # Proper response from server
            data = response.read()
            response_json = json.loads(data.decode("utf-8"))
            if len(response_json) > 0:
                data_size_kb = data.__sizeof__() / 1000
                print(
                    f"OSM: downloaded {data_size_kb:.1f} KB from {NOMINATIM_ENDPOINT}"
                )
                # We only return top of the data
                retrieved_data = response_json
            else:
                raise RuntimeError(f"OSM: no matching data for {query}!")
        except HTTPError as e:

            if e.code in [429, 504]:
                # 429 is 'too many requests' and 504 is 'gateway timeout' from
                # server overload: handle these by pausing then recursively
                # re-trying until we get a valid response from the server
                f"OSM: {NOMINATIM_ENDPOINT} returned {sc}: retry in 30 secs"
                time.sleep(30)
                continue
            else:
                # else, this was an unhandled status code, throw an exception
                raise RuntimeError(
                    f"OSM: {NOMINATIM_ENDPOINT} returned {e.code}: {e.reason}!"
                )

    return retrieved_data


def get_filter():
    pass
