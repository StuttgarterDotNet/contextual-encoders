"""
MatrixComputer
====================================
The *MatrixComputer* combines the :class:`.Measure` with the :class:`.Gatherer` and calculates the similarity or
dissimilarity matrix for one attribute.
Thus, the *MatrixComputer* can be seen as a mapping :math:`\\mathcal{M}: F \\rightarrow \\mathbb{R}^{n \\times n}`,
with :math:`F` being the feature space and :math:`n` the amount of features.
"""

import numpy as np
from .gatherer import GathererFactory, Gatherer


class MatrixComputer:
    """
    The service class to compute a similarity or dissimilarity matrix.
    """

    def __init__(self, measure, gatherer, separator_token):
        """
        Initializes the *MatrixComputer*.

        :param measure: The instance of the *Similarity* or *Dissimilarity Measure*.
            See :class:`.SimilarityMeasure` and :class:`.DissimilarityMeasure`.
        :param gatherer: Either the name of a *Gatherer* or a concrete instance.
            See :class:`.GathererFactory` for implemented *Gatherers* and
            :class:`.Gatherer` for creating custom *Gatherers*.
            If the specified measure can handle multiple values (forms of an attribute),
            the :class:`.IdentityGatherer` will be taken in any way.
        :param separator_token: A string for separating forms of categorical attributes.
        """
        self.__measure = measure
        self.__separator_token = separator_token

        if self.__measure.can_handle_multiple_values():
            self.__gatherer = GathererFactory.create("id")
        else:
            if isinstance(gatherer, Gatherer):
                self.__gatherer = gatherer
            else:
                self.__gatherer = GathererFactory.create(gatherer)

    def compute(self, data):
        """
        Computes the similarity or dissimilarity matrix based on the given data.

        :param data: A single pandas series containing the data.
            Note, that each entry can have multiple values (the forms of an attribute),
            that are separated with the ``separator_token``.
        :return: A 2D numpy array representing the similarity or dissimilarity matrix.
        """
        n_samples = len(data)
        matrix = np.zeros((n_samples, n_samples))

        for i in range(0, n_samples):
            for j in range(0, n_samples):
                first = str(data[i])
                first = first.split(self.__separator_token)
                second = str(data[j])
                second = second.split(self.__separator_token)
                self.__gatherer.set_measure(self.__measure)
                matrix[i, j] = self.__gatherer.gather(first, second)

        return matrix
