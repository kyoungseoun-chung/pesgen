#!/usr/bin/env python3
"""Test downloader.py"""
import numpy as np

from pesgen.downloader import osm_network
from pesgen.downloader import query_to_gd
from pesgen.graph import Graph


def test_get_data_from_nominatim():
    location = "Grange, Edinburgh, Scotland"

    data = query_to_gd(location)
    polygon = data["geometry"]
    p_utm = polygon.copy().to_utm()
    b_polygon = p_utm.copy().buffered(500)
    b_polygon = b_polygon.copy().to_lnglat()

    network = osm_network(b_polygon)
    graph = Graph(network)
    graph.truncate(polygon)
    graph.count_streets_per_node()
    graph.simplify_graph()

    _, data_xy = zip(*graph.nx.nodes(data=True))

    nodes = np.asarray([[d["x"], d["y"]] for d in data_xy], dtype=np.float64)

    import matplotlib.pyplot as plt

    plt.scatter(nodes[:, 0], nodes[:, 1])
    plt.show()

    pass
