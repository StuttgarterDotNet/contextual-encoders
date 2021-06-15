"""
Measure
====================================
A *Measure* is used to calculate a comparison value between two attribute forms.

Let :math:`x, y \\in F` be two features from the feature space :math:`F`.
Each feature consists of :math:`k` attributes and each attribute can have up to :math:`l` forms.
A form of an attribute of the feature :math:`x` can then be denoted as :math:`x_{a,i}`,
with :math:`a` being the attribute and :math:`i` being the form.
For simplicity, we just denote it as :math:`\\tilde{x}_i` and skip the attribute index.
A *Measure* is then defined as :math:`\\mathcal{M} : (\\tilde{x}_i, \\tilde{y}_j) \\rightarrow [0,1]`, i.e.
it maps a similarity or dissimilarity value to each attribute form.

.. note::

    A *Measure* can also be defined on an attribute, rather then on attribute forms.
    This can be done by setting ``multiple_values`` to ``True``, see :class:`.Measure`.

.. note::

    A *Measure* always needs to return values within the range :math:`[0,1]`.
"""

import json
import networkx as nx
from networkx.algorithms.dag import dag_longest_path
from abc import ABC, abstractmethod


class Measure(ABC):
    """
    The abstract base class for all implementations of *Measures*.
    """

    def __init__(self, symmetric, multiple_values):
        """
        Initializes the *Measure*.

        :param symmetric: Defines whether the *Measure* is symmetric,
            i.e. if :math:`\\mathcal{M}(x,y) = \\mathcal{M}(y,x)`.
        :param multiple_values: Defines whether the *Measure* can compare full attributes,
            rather then only attribute forms. When this property is set to ``True``,
            the ``_compare`` method will get the entire attribute value as input. If
            the property is set to ``False``, a list with all attribute forms will be given
            as input.
        """
        self.__symmetric = symmetric
        self.__multiple_values = multiple_values
        self.__cache = dict()

    @abstractmethod
    def _compare(self, first, second):
        """
        Compares the two attributes or attribute forms. This is the abstract method that
        needs to be implemented by concrete *Measure* instances.

        :param first: The first attribute or attribute form.
        :param second: The second attribute or attribute form.
        :return: The comparison value which is in :math:`[0,1]`.
        """
        pass

    def compare(self, first, second):
        """
        Compares the two attributes or attribute forms.
        This method caches precalculated values within an in-memory dictionary.

        :param first: The first attribute or attribute form.
        :param second: The second attribute or attribute form.
        :return: The comparison value which is in :math:`[0,1]`.
        """
        cached_value = self.__read_from_cache(first, second)
        if cached_value is not None:
            return cached_value
        else:
            value = self._compare(first, second)
            self.__write_to_cache(first, second, value)
            return value

    @staticmethod
    def __generate_cache_key(first, second):
        """
        Generates a serializable cache key given the two attributes or attribute forms.

        :param first: The first attribute or attribute form.
        :param second: The second attribute or attribute form.
        :return: A unique key representing the two attributes or attribute forms.
        """
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
        """
        Writes the comparison value with the two attributes or attribute forms into the cache.
        :param first: The first attribute or attribute form.
        :param second: The second attribute or attribute form.
        :param value: The comparison value.
        """
        cache_key = self.__generate_cache_key(first, second)

        self.__cache[cache_key] = value

        return

    def __read_from_cache(self, first, second):
        """
        Read the comparison value for the two given attributes or attribute forms from the cache.
        If the value cannot be found in the cache, ``None`` will be returned instead.

        :param first: The first attribute or attribute form.
        :param second: The second attribute or attribute form.
        :return: The comparison value or ``None`` if it cannot be found.
        """
        cache_key = self.__generate_cache_key(first, second)
        reverse_cache_key = self.__generate_cache_key(second, first)

        if cache_key in self.__cache:
            return self.__cache[cache_key]
        elif self.__symmetric and reverse_cache_key in self.__cache:
            return self.__cache[reverse_cache_key]
        else:
            return None

    def is_symmetric(self):
        """
        Returns ``True`` if the *Measure* is symmetric, i.e. if :math:`\\mathcal{M}(x,y) = \\mathcal{M}(y,x)`.

        :return: ``True`` if the *Measure* is symmetric.
        """
        return self.__symmetric

    def can_handle_multiple_values(self):
        """
        Returns ``True`` if the *Measure* can handle multiple values.
        When this property is set to ``True``, the ``_compare`` method
        will get the entire attribute value as input. If the property is
        set to ``False``, a list with all attribute forms will be given
        as input.

        :return: ``True`` if the *Measure* can handle multiple values.
        """
        return self.__multiple_values

    def export_to_file(self, path):
        """
        Exports the *Measure* including the cache to the given path.

        :param path: The path to export the *Measure* to.
        """
        with open(path, "w") as file:
            json.dump(self.__cache, file, indent=4)

        return

    def import_from_file(self, path):
        """
        Imports the *Measure* including the cache from the given path.

        :param path: The path to import the *Measure* from.
        """
        with open(path, "r") as file:
            self.__cache = json.load(file)

        return


class SimilarityMeasure(Measure, ABC):
    """
    An abstract base class for calculating similarity values.
    """

    def __init__(self, symmetric, multiple_values):
        """
        Initializes the *Similarity Measure*.

        :param symmetric: Defines whether the *Similarity Measure* is symmetric,
            i.e. if :math:`\\mathcal{M}(x,y) = \\mathcal{M}(y,x)`.
        :param multiple_values: Defines whether the *Similarity Measure* can compare full attributes,
            rather then only attribute forms. When this property is set to ``True``,
            the ``_compare`` method will get the entire attribute value as input. If
            the property is set to ``False``, a list with all attribute forms will be given
            as input.
        """
        super().__init__(symmetric=symmetric, multiple_values=multiple_values)


class DissimilarityMeasure(Measure, ABC):
    """
    An abstract base class for calculating dissimilarity values.
    """

    def __init__(self, symmetric, multiple_values):
        """
        Initializes the *Dissimilarity Measure*.

        :param symmetric: Defines whether the *Dissimilarity Measure* is symmetric,
            i.e. if :math:`\\mathcal{M}(x,y) = \\mathcal{M}(y,x)`.
        :param multiple_values: Defines whether the *Dissimilarity Measure* can compare full attributes,
            rather then only attribute forms. When this property is set to ``True``,
            the ``_compare`` method will get the entire attribute value as input. If
            the property is set to ``False``, a list with all attribute forms will be given
            as input.
        """
        super().__init__(symmetric=symmetric, multiple_values=multiple_values)


class WuPalmer(SimilarityMeasure):
    """
    A tree based similarity measure based on the Wu-Palmer Similarity Measure.
    """

    def __init__(self, context, offset=0.0):
        """
        Initializes the *WuPalmer Similarity Measure*.

        :param context: The :class:`.TreeContext` used for comparison.
        :param offset: Either a real value or ``depth``. If a real value is used,
            the distance between the root and a concept will always be at least the
            value of the offset. If ``depth`` is used, the offset will be :math:`\\frac{1}{N}`,
            with :math:`N` being the depth of the tree. Using an offset prevent from getting
            a zero similarity.
        """
        super().__init__(symmetric=True, multiple_values=False)

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
        """
        Compares the two given attribute forms using the *WuPalmer Similarity Measure*.
        :param first: The first attribute form.
        :param second: The second attribute form.
        :return: The *WuPalmer Similarity* comparison value.
        """
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
            nx.algorithms.shortest_paths.generic.shortest_path_length(ud_graph, lca, root)
            + self.__offset
        )

        # if first and second, both is the root
        if d1 + d2 + 2.0 * d3 == 0.0:
            return 0.0

        return 2.0 * d3 / (d1 + d2 + 2.0 * d3)


class PathLengthMeasure(SimilarityMeasure):
    """
    A *SimilarityMeasure* based on counting the path length between two concepts.
    """

    def __init__(self, context):
        """
        Initializes the *PathLengthMeasure*.

        :param context: The :class:`.GraphContext` used for comparison.
        """
        super().__init__(symmetric=True, multiple_values=False)
        self.__context = context

        return

    def _compare(self, first, second):
        """
        Compares the two attribute forms based on their path length in the *Context*.
        The *Measure* counts the shortest path length :math:`p` going from the first to the second value
        and returns :math:`\\frac{1}{1+p}`.

        :param first: The first attribute form.
        :param second: The second attribute form.
        :return: The *PathLength Similarity* comparison value.
        """
        graph = self.__context.get_graph().to_undirected()
        shortest_path_length = nx.shortest_path_length(graph, first, second)

        return 1.0 / (1.0 + shortest_path_length)
