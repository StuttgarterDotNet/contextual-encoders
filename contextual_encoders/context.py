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


class TreeContext(Context):
    def __init__(self, name):
        super().__init__(name)
        self.__graph = nx.DiGraph()

    def export_to_file(self, path):
        with open(path, "w") as file:
            file.write(json.dumps(nx.readwrite.json_graph.node_link_data(self.__graph)))
        return

    def import_from_file(self, path):
        with open(path, "r") as file:
            self.__graph = nx.readwrite.json_graph.node_link_data(json.load(file))
        return

    def add_concept(self, child, parent=None, weight=1.0):
        if parent is None:
            parent = self._name

        if not self.__graph.has_node(parent):
            self.__graph.add_node(parent)
        if not self.__graph.has_node(child):
            self.__graph.add_node(child)
        if not self.__graph.has_edge(parent, child):
            self.__graph.add_edge(parent, child, weight=weight)
        elif self.__graph.get_edge_data(parent, child)["weight"] is not weight:
            self.__graph.remove_edge(parent, child)
            self.__graph.add_edge(parent, child, weight=weight)

    def get_tree(self):
        return self.__graph

    def get_root(self):
        return self._name

    def draw(self):
        nx.draw(self.__graph)
        plt.show()
