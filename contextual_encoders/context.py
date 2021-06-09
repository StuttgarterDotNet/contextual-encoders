"""
Context
====================================
The *Context* is the core part of the Contextual Encoders library.
It is used to measure the similarity or dissimilarity of attributes.
So far, two different Context-Types are implemented: *GraphContext* and *TreeContext*.
However, it is very likely that custom context needs to be implemented.
Therefore, the base classes *Context* and *GraphBasedContext* are used,
that come with optimized in- and export functions as well as caching.
"""

from abc import ABC, abstractmethod
import networkx as nx
import json
import matplotlib.pyplot as plt


class Context(ABC):
    """
    The abstract base class for all Context.
    """
    def __init__(self, name):
        """
        Initializes the Context.

        :param name: The name of the Context.
        """
        self._name = name

        return

    @abstractmethod
    def export_to_file(self, path):
        """
        Exports the Context to the given file path.

        :param path: The path to export the Context to.
        """
        pass

    @abstractmethod
    def import_from_file(self, path):
        """
        Imports the Context from the given file path.

        :param path: The path to import the Context from.
        """
        pass


class GraphBasedContext(Context):
    """
    A base class for all graph based Context.
    """
    def __init__(self, name):
        """
        Initializes the GraphBasedContext.

        :param name: The name of the Context.
        """
        super().__init__(name)
        self._graph = nx.DiGraph()

        return

    def export_to_file(self, path):
        """
        Exports the graph to the given file path.

        :param path: The path to export the graph to.
        """
        with open(path, "w") as file:
            file.write(json.dumps(nx.readwrite.json_graph.node_link_data(self._graph)))

        return

    def import_from_file(self, path):
        """
        Imports the graph from the given file path.

        :param path: The path to import the graph from.
        """
        with open(path, "r") as file:
            self._graph = nx.readwrite.json_graph.node_link_data(json.load(file))

        return

    def get_graph(self):
        """
        Returns the networkx DiGraph instance.

        :return: A networkx DiGraph instance.
        """
        return self._graph

    def draw(self):
        """
        Draws the graph using matplotlib.
        """
        print(self._graph.edges)
        nx.draw(self._graph, with_labels=True)
        plt.show()

        return


# noinspection DuplicatedCode
class GraphContext(GraphBasedContext):
    """
    A graph based Context than can be used for graph based measures.
    """
    def add_concept(self, node, neighbor=None, weight=1.0):
        """
        Adds a new node to the graph.
        If the neighbor does not exist, it will be added as new node.
        If the node already exists, the weight will be overwritten.

        :param node: The name of the node.
        :param neighbor: The name of the neighbor node.
        :param weight: The wight of the edge between the node and the neighbor.
        """
        if not self._graph.has_node(node):
            self._graph.add_node(node)

        if neighbor is not None:
            if not self._graph.has_node(neighbor):
                self._graph.add_node(neighbor)
            if not self._graph.has_edge(node, neighbor):
                self._graph.add_edge(node, neighbor, weight=weight)
            elif self._graph.get_edge_data(node, neighbor)["weight"] is not weight:
                self._graph.remove_edge(node, neighbor)
                self._graph.add_edge(node, neighbor, weight=weight)

        return


# noinspection DuplicatedCode
class TreeContext(GraphBasedContext):
    """
    A graph based Context than can be used for tree based measures.
    """
    def add_concept(self, child, parent=None, weight=1.0):
        """
        Adds a new node to the tree, where the name of the context serves as the root node.
        If the parent does not exist, it will be added as new node.
        If the parent is None, the root node will serve as the parent.
        If the node already exists, the weight will be overwritten.

        :param child: The name of the child node.
        :param parent: The name of the parent node.
        :param weight: The wight of the edge between the child and the parent.
        """
        if parent is None:
            parent = self._name

        if not self._graph.has_node(parent):
            self._graph.add_node(parent)
        if not self._graph.has_node(child):
            self._graph.add_node(child)
        if not self._graph.has_edge(parent, child):
            self._graph.add_edge(parent, child, weight=weight)
        elif self._graph.get_edge_data(parent, child)["weight"] is not weight:
            self._graph.remove_edge(parent, child)
            self._graph.add_edge(parent, child, weight=weight)

        return

    def get_tree(self):
        """
        Returns the networkx DiGraph instance.

        :return: A networkx DiGraph representing the tree.
        """
        return self._graph

    def get_root(self):
        """
        Gets the name of the root, i.e. the name of the context.

        :return: The name of the root.
        """
        return self._name
