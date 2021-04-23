import networkx as nx
import matplotlib.pyplot as plt


class TreeContext:

    def __init__(self, name):
        self.__name = name
        self.__graph = nx.DiGraph()

    def add_concept(self, child, parent=None, weight=1.0):
        if parent is None:
            parent = self.__name

        if not self.__graph.has_node(parent):
            self.__graph.add_node(parent)
        if not self.__graph.has_node(child):
            self.__graph.add_node(child)
        if not self.__graph.has_edge(parent, child):
            self.__graph.add_edge(parent, child, weight=weight)
        elif self.__graph.get_edge_data(parent, child)['weight'] is not weight:
            self.__graph.remove_edge(parent, child)
            self.__graph.add_edge(parent, child, weight=weight)

    def get_tree(self):
        return self.__graph

    def get_root(self):
        return self.__name

    def draw(self):
        nx.draw(self.__graph)
        plt.show()
