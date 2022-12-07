#!/usr/bin/env python3
"""Graph from OSM network data."""
from dataclasses import dataclass
from datetime import date

from networkx import MultiDiGraph

from pesgen import __version__

TODAY = (date.today()).strftime("%d/%m/%Y")


@dataclass
class Graph(MultiDiGraph):

    osm_network: dict

    def __post_init__(self):

        assert "elements" in self.osm_network, "Graph: empty network data!"

        metadata = {
            "created_date": TODAY,
            "created_with": f"pesgen {__version__}",
            "crs": "EPSG:4326",
        }

        super().__init__(**metadata)


def graph_from_place(query: str):
    pass
