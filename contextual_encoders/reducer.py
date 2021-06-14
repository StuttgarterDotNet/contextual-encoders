"""
Reducer
====================================
A *Reducer* transforms a similarity or dissimilarity matrix into a set of vectors.
Mathematically, it can be seen as
a map :math:`\\mathcal{R} : D \\in \\mathbb{R}^{n \\times n} \\rightarrow  \\tilde{X} \\subset \\mathbb{R}^{m}`,
with :math:`m \\in \mathbb{N}` being the (configurable) dimension of the encoding
and :math:`\\tilde{X}` the encoded dataset as vectors.

In other words, let :math:`n \\in \\mathbb{N}` be the amount of features.
A *Reducer* then takes the similarity or dissimilarity matrix :math:`D \\in \\mathbb{R}^{n \\times n}`
and produces :math:`n` euclidean vectors of dimension :math:`m`.
"""

from abc import ABC, abstractmethod
from sklearn.manifold import MDS

MultidimensionalScaling = "mds"


class Reducer(ABC):
    """
    The abstract base class for all reducers.
    """

    def __init__(self, n_components):
        """
        Initializes the reducer.

        :param n_components: The dimension of the output vectors.
        """
        self.__n_components = n_components

    @abstractmethod
    def reduce(self, matrix):
        """
        The abstract method that is implemented by concrete instances of reducers.

        :param matrix: The similarity or dissimilarity matrix
            :math:`D \\in \\mathbb{R}^{n \\times n}` as 2D numpy array.
        :return: The set of vectors :math:`\\tilde{X} \\in \\mathbb{R}^{n \\times m}`,
            with :math:`m` being n_components.
        """
        pass


class SimilarityMatrixReducer(Reducer, ABC):
    """
    An abstract base class for reducing similarity matrices.
    """


class DissimilarityMatrixReducer(Reducer, ABC):
    """
    An abstract base class for reducing dissimilarity matrices.
    """


class ReducerFactory:
    """
    The factory class for creating reducers.
    """

    @staticmethod
    def create(reducer_type, **kwargs):
        """
        Creates a concrete reducer instance given the name.

        :param reducer_type: The name of the reducer. Currently, only the ``mds`` reducer is implemented.
        :param kwargs:
        :return:
        """
        if "n_components" not in kwargs:
            kwargs["n_components"] = 2
        if "metric" not in kwargs:
            kwargs["metric"] = True

        if reducer_type == MultidimensionalScaling:
            return MultidimensionalScalingReducer(
                n_components=kwargs["n_components"], metric=kwargs["metric"]
            )
        else:
            raise ValueError(f"A reducer of type {reducer_type} does not exist.")


class MultidimensionalScalingReducer(DissimilarityMatrixReducer):
    """
    A reducer using the
    `Multidimensional Scaling <https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html>`_
    approach (MDS) from scikit-learn.
    """

    def __init__(self, n_components, metric):
        """
        Initializes the *MultidimensionalScalingReducer*.

        :param n_components: The dimension of the output vectors.
        :param metric:
        """
        super().__init__(n_components)
        self.__mds = MDS(n_components, metric=metric, dissimilarity="precomputed")

    def reduce(self, matrix):
        return self.__mds.fit_transform(matrix)

    def get_stress(self):
        return self.__mds.stress_
