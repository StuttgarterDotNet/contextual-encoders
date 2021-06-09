"""
Measure
====================================
A *Gatherer* is used to combine a set of pairwise attribute measures to a single measure.

.. note::

    If a measure can handle multiple values, a Gatherer is not needed.
"""

import json
import networkx as nx
from networkx.algorithms.dag import dag_longest_path
from abc import ABC, abstractmethod


class Measure(ABC):
    def __init__(self, symmetric, multiple_values, verbose=False):
        self.__symmetric = symmetric
        self.__multiple_values = multiple_values
        self.__verbose = verbose
        self.__cache = dict()

    @abstractmethod
    def _compare(self, first, second):
        pass

    def compare(self, first, second):
        cached_value = self.__read_from_cache(first, second)
        if cached_value is not None:
            return cached_value
        else:
            value = self._compare(first, second)
            self.__write_to_cache(first, second, value)
            return value

    @staticmethod
    def __generate_cache_key(first, second):
        first_string = first
        second_string = second

        # if we have list or tuple, convert to comma separated string
        if isinstance(first_string, list) or isinstance(first_string, tuple):
            first_string = ",".join(first_string)
        if isinstance(second_string, list) or isinstance(second_string, tuple):
            second_string = ",".join(second_string)

        # if we don't have a string now, convert to hash-string then tuple
        if not isinstance(first_string, str):
            first_string = str(hash(first_string))
        if not isinstance(second_string, str):
            second_string = str(hash(second_string))

        return str((first_string, second_string))

    def __write_to_cache(self, first, second, value):
        cache_key = self.__generate_cache_key(first, second)

        if self.__verbose:
            print(f'Add "{cache_key}" to cache of "{type(self).__name__}".')
        self.__cache[cache_key] = value

        return

    def __read_from_cache(self, first, second):
        cache_key = self.__generate_cache_key(first, second)
        reverse_cache_key = self.__generate_cache_key(second, first)

        if cache_key in self.__cache:
            return self.__cache[cache_key]
        elif self.__symmetric and reverse_cache_key in self.__cache:
            return self.__cache[reverse_cache_key]
        else:
            return None

    def is_symmetric(self):
        return self.__symmetric

    def can_handle_multiple_values(self):
        return self.__multiple_values

    def export_to_file(self, path):
        with open(path, "w") as file:
            json.dump(self.__cache, file, indent=4)

        return

    def import_from_file(self, path):
        with open(path, "r") as file:
            self.__cache = json.load(file)

        return


class SimilarityMeasure(Measure, ABC):
    def __init__(self, symmetric, multiple_values, verbose=False):
        super().__init__(
            symmetric=symmetric, multiple_values=multiple_values, verbose=verbose
        )


class DissimilarityMeasure(Measure, ABC):
    def __init__(self, symmetric, multiple_values, verbose=False):
        super().__init__(
            symmetric=symmetric, multiple_values=multiple_values, verbose=verbose
        )


class WuPalmer(SimilarityMeasure):
    def __init__(self, context, offset=0.0, verbose=False):
        super().__init__(symmetric=True, multiple_values=False, verbose=verbose)

        self.__context = context

        if isinstance(offset, str):
            if offset == "depth":
                graph = self.__context.get_tree()
                depth = len(dag_longest_path(graph))
                self.__offset = 1.0 / depth
            else:
                raise ValueError(
                    "The value " + str(offset) + ' is not valid for "offset".'
                )
        else:
            self.__offset = offset + 0.0

        return

    def _compare(self, first, second):
        # get directed graph
        d_graph = self.__context.get_tree()

        # get undirected graph
        ud_graph = d_graph.to_undirected()

        # get lowest reachable node from both
        lca = nx.algorithms.lowest_common_ancestors.lowest_common_ancestor(
            d_graph, first, second
        )

        # get root of graph
        root = self.__context.get_root()

        # count edges
        d1 = nx.algorithms.shortest_paths.generic.shortest_path_length(
            ud_graph, first, lca
        )
        d2 = nx.algorithms.shortest_paths.generic.shortest_path_length(
            ud_graph, second, lca
        )
        d3 = (
            nx.algorithms.shortest_paths.generic.shortest_path_length(
                ud_graph, lca, root
            )
            + self.__offset
        )

        # if first and second, both is the root
        if d1 + d2 + 2.0 * d3 == 0.0:
            return 0.0

        return 2.0 * d3 / (d1 + d2 + 2.0 * d3)


class PathLengthMeasure(SimilarityMeasure):
    def __init__(self, context, verbose=False):
        super().__init__(symmetric=True, multiple_values=False, verbose=verbose)
        self.__context = context

        return

    def _compare(self, first, second):
        graph = self.__context.get_graph().to_undirected()
        shortest_path_length = nx.shortest_path_length(graph, first, second)

        return 1.0 / (1.0 + shortest_path_length)
