#!/usr/bin/env python3
"""Test downloader.py"""
from pesgen.downloader import osm_network
from pesgen.downloader import Polygon
from pesgen.downloader import query_to_gd
from pesgen.graph import Graph


def test_get_data_from_nominatim():
    location = "Grange, Edinburgh, Scotland"

    import matplotlib.pyplot as plt

    data = query_to_gd(location)
    polygon: Polygon = data["geometry"]
    p_utm = polygon.copy().to_utm()
    b_polygon = p_utm.copy().buffered(500)
    b_polygon = b_polygon.copy().to_lnglat()

    network = osm_network(polygon)
    graph = Graph(network)
    graph.truncate(polygon)
    # graph.count_streets_per_node()
    # graph.simplify_graph()

    plt.plot(polygon.coord[0], polygon.coord[1], "b")
    plt.plot(b_polygon.coord[0], b_polygon.coord[1], "r")
    plt.scatter(graph.coord[:, 0], graph.coord[:, 1], s=10, color="k")

    plt.show()

    # plt.scatter(nodes[:, 0], nodes[:, 1])
    # plt.show()

    pass
