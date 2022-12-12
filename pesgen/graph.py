#!/usr/bin/env python3
"""Graph from OSM network data.
Note:
    - Majority of code is from `osmnx` package.
    - Author only made minor modification to integrate `osmnx` features into `Graph` class.
        - We do not use `geopandas` module.
        - Some options that are not in use of our purpose are ignored.
        - Type added.
"""
import itertools
import warnings
from collections import Counter
from collections.abc import Generator
from dataclasses import dataclass
from datetime import date
from functools import cached_property

import networkx
import numpy as np
from networkx import MultiDiGraph
from numpy.typing import NDArray
from shapely.geometry import box
from shapely.geometry import LineString
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.ops import split
from shapely.strtree import STRtree

from pesgen import __version__
from pesgen.geometry import Polygon

TODAY = (date.today()).strftime("%d/%m/%Y")
TAG_NODES = ["ref", "highway"]
TAG_PATHS = [
    "bridge",
    "tunnel",
    "oneway",
    "lanes",
    "ref",
    "name",
    "highway",
    "maxspeed",
    "service",
    "access",
    "area",
    "landuse",
    "width",
    "est_width",
    "junction",
]

METADATA: dict[str, str] = {
    "created_date": TODAY,
    "created_with": f"pesgen {__version__}",
    "crs": "EPSG:4326",
}
EARTH_RADIUS_M: float = 6_371_009

LL_TYPE = NDArray[np.float64]

QUADRAT_WIDTH: float = 0.05
MIN_NUM: int = 3


@dataclass
class Graph:

    osm_network: dict
    nx_graph: MultiDiGraph = MultiDiGraph(**METADATA)

    def __post_init__(self):

        assert "elements" in self.osm_network, "Graph: empty network data!"

        nodes, paths = self.parse_node_path()

        node_test = np.asarray(
            [[nodes[d]["x"], nodes[d]["y"]] for d in nodes], dtype=np.float64
        )

        # add each osm node to the graph
        for node, data in nodes.items():
            self.nx.add_node(node, **data)

        # add each osm way (ie, a path of edges) to the graph
        self.add_paths(list(paths.values()))

        # retain only the largest connected component if retain_all is False
        self.get_largest_component()

        # add length (great-circle distance between nodes) attribute to each edge
        if len(self.nx.edges) > 0:
            self.add_edge_lengths()

    @cached_property
    def coord(self) -> NDArray[np.float64]:
        """Return coordinates of nodes."""
        return np.asarray(
            [[n[1]["x"], n[1]["y"]] for n in self.nx.nodes(data=True)],  # type: ignore
            dtype=np.float64,
        )

    def add_paths(self, paths: list) -> None:
        """
        Add a list of paths to the graph as edges.
        Parameters

        Args:
            G (MultiDiGraph): graph to add paths to
            paths (list): list of paths' tag:value attribute data dicts
        """
        # the values OSM uses in its 'oneway' tag to denote True, and to denote
        # travel can only occur in the opposite direction of the node order. see:
        # https://wiki.openstreetmap.org/wiki/Key:oneway
        # https://www.geofabrik.de/de/data/geofabrik-osm-gis-standard-0.7.pdf
        oneway_values = {"yes", "true", "1", "-1", "reverse", "T", "F"}
        reversed_values = {"-1", "reverse", "T"}

        for path in paths:

            # extract/remove the ordered list of nodes from this path element so
            # we don't add it as a superfluous attribute to the edge later
            nodes = path.pop("nodes")

            # reverse the order of nodes in the path if this path is both one-way
            # and only allows travel in the opposite direction of nodes' order
            is_one_way = _is_path_one_way(path, oneway_values)
            if is_one_way and _is_path_reversed(path, reversed_values):
                nodes.reverse()

            path["oneway"] = is_one_way

            # zip path nodes to get (u, v) tuples like [(0,1), (1,2), (2,3)].
            edges = list(zip(nodes[:-1], nodes[1:]))

            # add all the edge tuples and give them the path's tag:value attrs
            path["reversed"] = False
            self.nx.add_edges_from(edges, **path)

            # if the path is NOT one-way, reverse direction of each edge and add
            # this path going the opposite direction too
            if not is_one_way:
                path["reversed"] = True
                self.nx.add_edges_from([(v, u) for u, v in edges], **path)

    def simplify_graph(self):
        """
        Simplify a graph's topology by removing interstitial nodes.
        Simplifies graph topology by removing all nodes that are not intersections
        or dead-ends. Create an edge directly between the end points that
        encapsulate them, but retain the geometry of the original edges, saved as
        a new `geometry` attribute on the new edge. Note that only simplified
        edges receive a `geometry` attribute. Some of the resulting consolidated
        edges may comprise multiple OSM ways, and if so, their multiple attribute
        values are stored as a list.
        Parameters

        Note:
            - Always in strict mode.
        """
        if "simplified" in self.nx.graph and self.nx.graph["simplified"]:
            raise RuntimeError(
                "Graph: This graph has already been simplified, cannot simplify it again."
            )

        # define edge segment attributes to sum upon edge simplification
        attrs_to_sum = {"length", "travel_time"}

        # make a copy to not mutate original graph object caller passed in
        G: MultiDiGraph = self.nx.copy()
        all_nodes_to_remove = []
        all_edges_to_add = []

        # generate each path that needs to be simplified
        for path in _get_paths_to_simplify(G):

            # add the interstitial edges we're removing to a list so we can retain
            # their spatial geometry
            path_attributes = dict()
            for u, v in zip(path[:-1], path[1:]):

                # there should rarely be multiple edges between interstitial nodes
                # usually happens if OSM has duplicate ways digitized for just one
                # street... we will keep only one of the edges (see below)
                edge_count = G.number_of_edges(u, v)
                if edge_count != 1:
                    warnings.warn(
                        f"Graph: Found {edge_count} edges between {u} and {v} when simplifying"
                    )

                # get edge between these nodes: if multiple edges exist between
                # them (see above), we retain only one in the simplified graph
                edge_data = G.edges[u, v, 0]
                for attr in edge_data:
                    if attr in path_attributes:
                        # if this key already exists in the dict, append it to the
                        # value list
                        path_attributes[attr].append(edge_data[attr])
                    else:
                        # if this key doesn't already exist, set the value to a list
                        # containing the one value
                        path_attributes[attr] = [edge_data[attr]]

            # consolidate the path's edge segments' attribute values
            for attr in path_attributes:
                if attr in attrs_to_sum:
                    # if this attribute must be summed, sum it now
                    path_attributes[attr] = sum(path_attributes[attr])
                elif len(set(path_attributes[attr])) == 1:
                    # if there's only 1 unique value in this attribute list,
                    # consolidate it to the single value (the zero-th):
                    path_attributes[attr] = path_attributes[attr][0]
                else:
                    # otherwise, if there are multiple values, keep one of each
                    path_attributes[attr] = list(set(path_attributes[attr]))

            # construct the new consolidated edge's geometry for this path
            path_attributes["geometry"] = LineString(
                [
                    Point((G.nodes[node]["x"], G.nodes[node]["y"]))
                    for node in path
                ]
            )

            # add the nodes and edge to their lists for processing at the end
            all_nodes_to_remove.extend(path[1:-1])
            all_edges_to_add.append(
                {
                    "origin": path[0],
                    "destination": path[-1],
                    "attr_dict": path_attributes,
                }
            )

        # for each edge to add in the list we assembled, create a new edge between
        # the origin and destination
        for edge in all_edges_to_add:
            G.add_edge(
                edge["origin"], edge["destination"], **edge["attr_dict"]
            )

        # finally remove all the interstitial nodes between the new edges
        G.remove_nodes_from(set(all_nodes_to_remove))

        # remove any connected components that form a self-contained ring
        # without any endpoints
        wccs = networkx.weakly_connected_components(G)
        nodes_in_rings = set()
        for wcc in wccs:
            if not any(_is_endpoint(G, n) for n in wcc):
                nodes_in_rings.update(wcc)
        G.remove_nodes_from(nodes_in_rings)

        # mark graph as having been simplified
        G.graph["simplified"] = True
        self.nx_graph = G

    def count_streets_per_node(self) -> None:
        """
        Count how many physical street segments connect to each node in a graph.
        This function uses an undirected representation of the graph and special
        handling of self-loops to accurately count physical streets rather than
        directed edges. Note: this function is automatically run by all the
        `graph.graph_from_x` functions prior to truncating the graph to the
        requested boundaries, to add accurate `street_count` attributes to each
        node even if some of its neighbors are outside the requested graph
        boundaries.
        """
        nodes = self.nx.nodes

        # get one copy of each self-loop edge, because bi-directional self-loops
        # appear twice in the undirected graph (u,v,0 and u,v,1 where u=v), but
        # one-way self-loops will appear only once
        Gu = self.nx.to_undirected(reciprocal=False, as_view=True)
        self_loop_edges = set(networkx.selfloop_edges(Gu))

        # get all non-self-loop undirected edges, including parallel edges
        non_self_loop_edges = [
            e for e in Gu.edges(keys=False) if e not in self_loop_edges
        ]

        # make list of all unique edges including each parallel edge unless the
        # parallel edge is a self-loop, in which case we don't double-count it
        all_unique_edges = non_self_loop_edges + list(self_loop_edges)

        # flatten list of (u, v) edge tuples to count how often each node appears
        edges_flat = itertools.chain.from_iterable(all_unique_edges)
        counts = Counter(edges_flat)
        spn = {node: counts[node] for node in nodes}
        networkx.set_node_attributes(self.nx, values=spn, name="street_count")

    def truncate(self, polygon: Polygon) -> None:
        """
        Remove every node in graph that falls outside a Polygon.
        Parameters

        Args:
            polygon (Polygon): pesgen.geometry.Polygon
            only retain nodes in graph that lie within this geometry
        """

        # first identify all nodes whose point geometries lie within the polygon
        osm_id, gs_nodes = _graph_to_gdf_nodes(self.nx)
        # Find all points inside the polygon
        to_keep = _intersect_index_quadrats(gs_nodes, osm_id, polygon)

        if to_keep.shape[0] == 0:
            # no graph nodes within the polygon: can't create a graph from that
            raise ValueError(
                "Graph: Found no graph nodes within the requested polygon"
            )

        # Now identify all nodes whose point geometries lie outside the polygon
        nodes_outside = np.setdiff1d(osm_id, to_keep)

        # Remove from the graph all those nodes that lie outside the polygon
        # make a copy to not mutate original graph object caller passed in
        self.nx_graph = self.nx.copy()
        self.nx_graph.remove_nodes_from(nodes_outside)

    def add_edge_lengths(self):
        """
        Add `length` attribute (in meters) to each edge.
        """
        uvk = tuple(self.nx.edges)

        # extract edge IDs and corresponding coordinates from their nodes
        x = self.nx.nodes(data="x")  # type: ignore
        y = self.nx.nodes(data="y")  # type: ignore

        try:
            # two-dimensional array of coordinates: y0, x0, y1, x1
            c = np.asarray([(y[u], x[u], y[v], x[v]) for u, v, _ in uvk])
        except KeyError:  # pragma: no cover
            raise KeyError(
                "Graph: some edges missing nodes, possibly due to input data clipping issue"
            )

        # calculate great circle distances, round, and fill nulls with zeros
        dists = _great_circle_vec(c[:, 0], c[:, 1], c[:, 2], c[:, 3]).round(3)
        dists[np.isnan(dists)] = 0
        networkx.set_edge_attributes(
            self.nx, values=dict(zip(uvk, dists)), name="length"
        )

    def get_largest_component(self) -> None:
        """
        Return the largest connected component of the graph.
        """
        is_connected = networkx.is_weakly_connected
        connected_components = networkx.weakly_connected_components

        if not is_connected(self.nx):
            # get all the connected components in graph then identify the largest
            largest_cc = max(connected_components(self.nx), key=len)

            # induce (frozen) subgraph then unfreeze it by making new MultiDiGraph
            self.nx_graph = MultiDiGraph(self.nx.subgraph(largest_cc))

    def parse_node_path(self) -> tuple[dict, dict]:
        """
        Construct dicts of nodes and paths from an Overpass response.
        Parameters

        Note: Copied and modified from `osmnx` package.
        """
        nodes = dict()
        paths = dict()
        for element in self.osm_network["elements"]:
            if element["type"] == "node":
                nodes[element["id"]] = _convert_node(element)
            elif element["type"] == "way":
                paths[element["id"]] = _convert_path(element)

        return nodes, paths

    @property
    def nx(self) -> MultiDiGraph:
        """Return `networkx` Graph object."""
        return self.nx_graph


def _get_paths_to_simplify(G: MultiDiGraph) -> Generator:
    """
    Generate all the paths to be simplified between endpoint nodes.
    The path is ordered from the first endpoint, through the interstitial nodes,
    to the second endpoint.
    Parameters

    Args:
        MultiDiGraph: input graph
    """
    # first identify all the nodes that are endpoints
    endpoints = {n for n in G.nodes if _is_endpoint(G, n)}

    # for each endpoint node, look at each of its successor nodes
    for endpoint in endpoints:
        for successor in G.successors(endpoint):
            if successor not in endpoints:
                # if endpoint node's successor is not an endpoint, build path
                # from the endpoint node, through the successor, and on to the
                # next endpoint node
                yield _build_path(G, endpoint, successor, endpoints)


def _build_path(
    G: MultiDiGraph, endpoint: int, endpoint_successor: int, endpoints: set
) -> list[int]:
    """
    Build a path of nodes from one endpoint node to next endpoint node.
    Parameters

    Args:
        G (MultiDiGraph): input graph
        endpoint (int): the endpoint node from which to start the path
        endpoint_successor (int): the successor of endpoint through which the
        endpoints (set): the set of all nodes in the graph that are endpoints

    Returns:
        list: the first and last items in the resulting path list are endpoint nodes, and all other items are interstitial nodes that can be removed subsequently
    """
    # start building path from endpoint node through its successor
    path = [endpoint, endpoint_successor]

    # for each successor of the endpoint's successor
    for successor in G.successors(endpoint_successor):
        if successor not in path:
            # if this successor is already in the path, ignore it, otherwise add
            # it to the path
            path.append(successor)
            while successor not in endpoints:
                # find successors (of current successor) not in path
                successors = [
                    n for n in G.successors(successor) if n not in path
                ]

                # 99%+ of the time there will be only 1 successor: add to path
                if len(successors) == 1:
                    successor = successors[0]
                    path.append(successor)

                # handle relatively rare cases or OSM digitization quirks
                elif len(successors) == 0:
                    if endpoint in G.successors(successor):
                        # we have come to the end of a self-looping edge, so
                        # add first node to end of path to close it and return
                        return path + [endpoint]
                    else:
                        # this can happen due to OSM digitization error where
                        # a one-way street turns into a two-way here, but
                        # duplicate incoming one-way edges are present
                        warnings.warn(
                            f"Graph: Unexpected simplify pattern handled near {successor}"
                        )
                        return path
                else:
                    # if successor has >1 successors, then successor must have
                    # been an endpoint because you can go in 2 new directions.
                    # this should never occur in practice
                    raise Exception(
                        f"Unexpected simplify pattern failed near {successor}"
                    )

            # if this successor is an endpoint, we've completed the path
            return path

    # if endpoint_successor has no successors not already in the path, return
    # the current path: this is usually due to a digitization quirk on OSM
    return path


def _is_endpoint(G: MultiDiGraph, node: int) -> bool:
    """
    Is node a true endpoint of an edge.
    Return True if the node is a "real" endpoint of an edge in the network,
    otherwise False. OSM data includes lots of nodes that exist only as points
    to help streets bend around curves. An end point is a node that either:
    1) is its own neighbor, ie, it self-loops.
    2) or, has no incoming edges or no outgoing edges, ie, all its incident
    edges point inward or all its incident edges point outward.
    3) or, it does not have exactly two neighbors and degree of 2 or 4.
    Parameters

    Args:
        G (MultiDiGraph): input graph
        node (int): the node to examine
    """
    neighbors = set(list(G.predecessors(node)) + list(G.successors(node)))
    n = len(neighbors)
    d = G.degree(node)

    # rule 1
    if node in neighbors:
        # if the node appears in its list of neighbors, it self-loops
        # this is always an endpoint.
        return True

    # rule 2
    elif G.out_degree(node) == 0 or G.in_degree(node) == 0:
        # if node has no incoming edges or no outgoing edges, it is an endpoint
        return True

    # rule 3
    elif not (n == 2 and (d == 2 or d == 4)):
        # else, if it does NOT have 2 neighbors AND either 2 or 4 directed
        # edges, it is an endpoint. either it has 1 or 3+ neighbors, in which
        # case it is a dead-end or an intersection of multiple streets or it has
        # 2 neighbors but 3 degree (indicating a change from oneway to twoway)
        # or more than 4 degree (indicating a parallel edge) and thus is an
        # endpoint
        return True

    # rule 4
    # if none of the preceding rules returned true, then it is not an endpoint
    else:
        return False


def _intersect_index_quadrats(
    geometries: list[Point], osm_id: NDArray[np.int64], polygon: Polygon
) -> NDArray[np.int64]:
    """
    Identify geometries that intersect a (multi)polygon.
    Uses an r-tree spatial index and cuts polygon up into smaller sub-polygons
    for r-tree acceleration. Ensure that geometries and polygon are in the
    same coordinate reference system.
    Parameters

    Note:
        - Unlike `osmnx` package, we directly use `shapely.strtree.STRtree` directly.

    Args:
        geometries (geopandas.GeoSeries): the geometries (Points) to intersect with the polygon
        osm_id (NDArray[np.int64]): the index of the graph nodes
        polygon (Polygon): the polygon to intersect with the geometries

    Returns
        set: index labels of geometries that intersected polygon
    """
    # create an r-tree spatial index for the geometries
    s_index = STRtree(geometries)
    # Shapely geometries are not hashable.
    index_by_id = {id(pt): i for i, pt in enumerate(geometries)}

    # cut the polygon into chunks for spatial index intersecting
    multipoly = _quadrat_cut_geometry(polygon)
    geoms_in_poly: set = set()

    # loop through each chunk of the polygon to find intersecting geometries
    for poly in multipoly.geoms:

        poly = poly.buffer(0.0)
        if poly.is_valid and poly.area > 0:

            # Approximated matches
            b_box = box(*poly.bounds)
            pts_in_box = s_index.query(b_box)
            osm_id_in_box = osm_id[pts_in_box]
            # Update tree with approximated match
            s_index_appx = STRtree(
                s_index.geometries.take(pts_in_box).tolist()
            )

            # Precise query
            geoms_in_poly.update(
                osm_id_in_box[s_index_appx.query(poly, predicate="intersects")]
            )

    return np.asarray(list(geoms_in_poly), dtype=np.int64)


def _quadrat_cut_geometry(polygon: Polygon) -> MultiPolygon:
    """
    Split a Polygon or MultiPolygon up into sub-polygons of a specified size.
    Parameters

    Args:
        polygon (Polygon): the geometry to split up into smaller sub-polygons

    Returns
        MultiPolygon
    """
    # create n evenly spaced points between the min and max x and y bounds
    west, south, east, north = polygon.geometry.bounds  # type: ignore
    x_num = int(np.ceil((east - west) / QUADRAT_WIDTH) + 1)
    y_num = int(np.ceil((north - south) / QUADRAT_WIDTH) + 1)
    x_points = np.linspace(west, east, num=max(x_num, MIN_NUM))
    y_points = np.linspace(south, north, num=max(y_num, MIN_NUM))

    # create a quadrat grid of lines at each of the evenly spaced points
    vertical_lines = [
        LineString([(x, y_points[0]), (x, y_points[-1])]) for x in x_points
    ]
    horizontal_lines = [
        LineString([(x_points[0], y), (x_points[-1], y)]) for y in y_points
    ]
    lines = vertical_lines + horizontal_lines

    # recursively split the geometry by each quadrat line
    polys = [polygon.geometry]

    for line in lines:
        # split polygon by line if they intersect, otherwise just keep it
        split_geoms = [
            split(p, line).geoms if p.intersects(line) else [p] for p in polys
        ]
        # now flatten the list and process these split geoms on the next line in the list of lines
        polys = [g for g_list in split_geoms for g in g_list]

    return MultiPolygon(polys)


def _graph_to_gdf_nodes(
    G: MultiDiGraph,
) -> tuple[NDArray[np.int64], list[Point]]:
    """
    Convert a MultiDiGraph to node and/or edge GeoDataFrames.
    This function is the inverse of `graph_from_gdfs`.
    Parameters

    Args:
        G: networkx.MultiDiGraph
    """
    if not G.nodes:  # pragma: no cover
        raise ValueError("graph contains no nodes")

    osm_id, data_xy = zip(*G.nodes(data=True))

    nodes: list[Point] = [Point(d["x"], d["y"]) for d in data_xy]  # type: ignore

    return np.asarray(osm_id, dtype=np.int64), nodes


def _great_circle_vec(
    lat1: LL_TYPE, lng1: LL_TYPE, lat2: LL_TYPE, lng2: LL_TYPE
) -> NDArray[np.float64]:
    """
    Calculate great-circle distances between pairs of points.
    Vectorized function to calculate the great-circle distance between two
    points' coordinates or between arrays of points' coordinates using the
    haversine formula. Expects coordinates in decimal degrees.
    Parameters

    Args:
        lat1 (NDArray[np.float64]): first point's latitude coordinate
        lng1 (NDArray[np.float64]): first point's longitude coordinate
        lat2 (NDArray[np.float64]): second point's latitude coordinate
        lng2 (NDArray[np.float64]): second point's longitude coordinate

    Returns:
        NDArray[np.float64]: distance from each (lat1, lng1) to each (lat2, lng2) in units of earth_radius
    """
    y1 = np.deg2rad(lat1)
    y2 = np.deg2rad(lat2)
    dy = y2 - y1

    x1 = np.deg2rad(lng1)
    x2 = np.deg2rad(lng2)
    dx = x2 - x1

    h = np.sin(dy / 2) ** 2 + np.cos(y1) * np.cos(y2) * np.sin(dx / 2) ** 2
    h = np.minimum(1, h)  # protect against floating point errors
    arc = 2.0 * np.arcsin(np.sqrt(h))

    # return distance in units of earth_radius
    return arc * EARTH_RADIUS_M


def _convert_node(element: dict) -> dict:
    """
    Convert an OSM node element into the format for a networkx node.
    Parameters

    Note: Copied from `osmnx` package.
    """
    node = {"y": element["lat"], "x": element["lon"]}
    if "tags" in element:
        for useful_tag in TAG_NODES:
            if useful_tag in element["tags"]:
                node[useful_tag] = element["tags"][useful_tag]
    return node


def _convert_path(element: dict) -> dict:
    """
    Convert an OSM way element into the format for a networkx path.
    Parameters

    Note: Copied from `osmnx` package.
    """
    path = {"osmid": element["id"]}

    # remove any consecutive duplicate elements in the list of nodes
    path["nodes"] = [group[0] for group in itertools.groupby(element["nodes"])]

    if "tags" in element:
        for useful_tag in TAG_PATHS:
            if useful_tag in element["tags"]:
                path[useful_tag] = element["tags"][useful_tag]
    return path


def _is_path_one_way(path: dict, oneway_values: set) -> bool:
    """
    Determine if a path of nodes allows travel in only one direction.
    Parameters

    Args:
        path (dict): a path's tag:value attribute data
        oneway_values (set): the values OSM uses in its 'oneway' tag to denote True

    Returns:
        bool
    """

    return (
        True
        if ("oneway" in path and path["oneway"] in oneway_values)
        or ("junction" in path and path["junction"] == "roundabout")
        else False
    )


def _is_path_reversed(path: dict, reversed_values: set) -> bool:
    """
    Determine if the order of nodes in a path should be reversed.
    Parameters

    Args:
        path (dict): a path's tag:value attribute data
        reversed_values (set): the values OSM uses in its 'oneway' tag to denote travel can only

    Returns:
        bool
    """

    return (
        True
        if "oneway" in path and path["oneway"] in reversed_values
        else False
    )


def graph_from_place(query: str):
    pass
