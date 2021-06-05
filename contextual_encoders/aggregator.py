"""
Aggregators
====================================
Aggregators are used to combine multiple matrices to a single matrix.
This is used to combine similarity or dissimilarity matrices of multiple attributes to a single one.
Thus, an aggregator :math:`\\mathcal{A}` is a mapping of the form
:math:`\\mathcal{A} : \\mathbb{R}^{n \\times n \\times m} \\rightarrow \\mathbb{R}^{n \\times n}`,
with :math:`n` being the amount of features and :math:`m` being the number of similarity or dissimilarity matrices of type :math:`D \\in \\mathbb{R}^{n \\times n}`.


Currently, the following aggregators are implement:

=========== ===========
Name        Formula
----------- -----------
Mean        :math:`\\mathcal{A} (D^1, D^2, ..., D^m) = \\frac{1}{m} \\sum_{i=1}^{m} D^i`
Median      :math:`\\mathcal{A} (D^1, D^2, ..., D^m) = \\left\{ \\begin{array}{ll} D^{\\frac{m}{2}}  & \\mbox{, if } m \\mbox{ is even} \\\\ \\frac{1}{2} \\left( D^{\\frac{m-1}{2}} + D^{\\frac{m+1}{2}} \\right) & \\mbox{, if } m \\mbox{ is odd} \\end{array} \\right.`
Max         :math:`\\mathcal{A} (D^1, D^2, ..., D^m) = max_{ k} \\; D_{i,j}^k`
Min         :math:`\\mathcal{A} (D^1, D^2, ..., D^m) = min_{ k} \\; D_{i,j}^k`
=========== ===========

Concrete instances of these aggregators can be created using the AggregatorFactory.
"""

import numpy as np
from abc import ABC, abstractmethod

Mean = "mean"
Median = "median"
Max = "max"
Min = "min"


class Aggregator(ABC):
    """
    An abstract base class for aggregators.
    """

    @abstractmethod
    def aggregate(self, matrices):
        """
        The abstract method that is implemented by the concrete aggregators.
        """
        pass


class AggregatorFactory:
    """
    The factory class for creating concrete instances of aggregators.
    """

    @staticmethod
    def create(aggregator):
        """
        Creates an instance of the given aggregator name.

        :param aggregator: name of the aggregator, which can be ``mean``, ``median``, ``max`` or ``min``.

        :return: an instance of the aggregator.

        :raise ValueError: The given aggregator does not exist.
        """
        if aggregator == Mean:
            return MeanAggregator()
        elif aggregator == Median:
            return MedianAggregator()
        elif aggregator == Max:
            return MaxAggregator()
        elif aggregator == Min:
            return MinAggregator()
        else:
            raise ValueError(f"An aggregator of type {aggregator} does not exist.")


class MeanAggregator(Aggregator):
    """
    This class aggregates similarity or dissimilarity matrices using the mean.
    """

    def aggregate(self, matrices):
        """
        Calculates :math:`D = \\mathcal{A} (D^1, D^2, ..., D^m) = \\frac{1}{m} \\sum_{i=1}^{m} D^i`.
        """
        return np.mean(matrices, axis=0)


class MedianAggregator(Aggregator):
    """
    lol
    """

    def aggregate(self, matrices):
        """
        lol
        """
        return np.median(matrices, axis=0)


class MaxAggregator(Aggregator):
    """
    lol
    """

    def aggregate(self, matrices):
        """
        lol
        """
        return np.max(matrices, axis=0)


class MinAggregator(Aggregator):
    """
    lol
    """

    def aggregate(self, matrices):
        """
        lol
        """
        return np.min(matrices, axis=0)
