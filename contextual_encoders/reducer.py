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

Currently, the following *Reducers* are implement:

=========== ===========
Name        Description
----------- -----------
mds         | Creates a low-dimensional representation of the data in which the distances respect well
            | the distances in the original high-dimensional space.
=========== ===========
"""

from abc import ABC, abstractmethod
from sklearn.manifold import MDS


class Reducer(ABC):
    """
    The abstract base class for all *Reducers*.
    """

    def __init__(self, n_components):
        """
        Initializes the *Reducer*.

        :param n_components: The dimension of the output vectors.
        """
        self.__n_components = n_components

    @abstractmethod
    def reduce(self, matrix):
        """
        The abstract method that is implemented by concrete instances of *Reducers*.

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
    The factory class for creating *Reducers* with default values.
    """

    @staticmethod
    def create(reducer):
        """
        Creates a concrete *Reducer* instance given the name.

        :param reducer: The name of the *Reducer*, which can be ``mds``.
        :return: The instance of the *Reducer*
        """
        if reducer == "mds":
            return MultidimensionalScalingReducer()
        else:
            raise ValueError(f"A reducer of type {reducer} does not exist.")


class MultidimensionalScalingReducer(DissimilarityMatrixReducer):
    """
    A reducer using the
    `Multidimensional Scaling <https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html>`_
    approach (MDS) from scikit-learn.
    It can be used with the ``mds`` option.
    """

    def __init__(self, n_components=2, metric=True):
        """
        Initializes the *MultidimensionalScalingReducer*.

        :param n_components: The dimension of the output vectors.
        :param metric: If ``True``, perform metric MDS; otherwise, perform nonmetric MDS.
        """
        super().__init__(n_components)
        self.__mds = MDS(n_components, metric=metric, dissimilarity="precomputed")

    def reduce(self, dissimilarity_matrix):
        """
        Reduces the given dissimilarity matrix using the MDS approach.

        :param dissimilarity_matrix: The dissimilarity matrix as 2D numpy array.
        :return: Encoded vectors as 2D numpy array of size :math:`n \\times m`,
            with :math:`n` being the amount of features
            and :math:`m` the dimension of the vectors, i.e. ``n_components``.
        """
        return self.__mds.fit_transform(dissimilarity_matrix)

    def get_stress(self):
        """
        Gets the stress level for the performed MDS.

        :return: The stress level of the MDS.
        """
        return self.__mds.stress_
