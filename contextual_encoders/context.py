from abc import ABC, abstractmethod
import networkx as nx
import json
import matplotlib.pyplot as plt


class Context(ABC):
    def __init__(self, name):
        self._name = name

    @abstractmethod
    def export_to_file(self, path):
        pass

    @abstractmethod
    def import_from_file(self, path):
        pass


class GraphBasedContext(Context):
    def __init__(self, name):
        super().__init__(name)
        self._graph = nx.DiGraph()

    def export_to_file(self, path):
        with open(path, "w") as file:
            file.write(json.dumps(nx.readwrite.json_graph.node_link_data(self._graph)))
        return

    def import_from_file(self, path):
        with open(path, "r") as file:
            self._graph = nx.readwrite.json_graph.node_link_data(json.load(file))
        return

    def get_graph(self):
        return self._graph

    def draw(self):
        print(self._graph.edges)
        nx.draw(self._graph, with_labels=True)
        plt.show()


# noinspection DuplicatedCode
class GraphContext(GraphBasedContext):
    def add_concept(self, node, neighbor=None, weight=1.0):
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
    def add_concept(self, child, parent=None, weight=1.0):
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
        return self._graph

    def get_root(self):
        return self._name
