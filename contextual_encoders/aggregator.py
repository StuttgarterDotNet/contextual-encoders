"""
Aggregator
====================================
*Aggregators* are used to combine multiple matrices to a single matrix.
This is used to combine similarity and dissimilarity matrices of multiple attributes to a single one.
Thus, an *Aggregator* :math:`\\mathcal{A}` is a mapping of the form
:math:`\\mathcal{A} : \\mathbb{R}^{n \\times n \\times k} \\rightarrow \\mathbb{R}^{n \\times n}`,
with :math:`n` being the amount of features and :math:`k` being the number of similarity or dissimilarity matrices
of type :math:`D \\in \\mathbb{R}^{n \\times n}`, i.e. the amount of attributes/columns of the dataset.


Currently, the following *Aggregators* are implement:

=========== ===========
Name        Formula
----------- -----------
mean        :math:`\\mathcal{A} (D^1, D^2, ..., D^k) = \\frac{1}{k} \\sum_{i=1}^{k} D^i`
median      :math:`\\mathcal{A} (D^1, D^2, ..., D^k) = \\left\\{ \\begin{array}{ll} D^{\\frac{k}{2}} & \\mbox{, if } k \\mbox{ is even} \\\\ \\frac{1}{2} \\left( D^{\\frac{k-1}{2}} + D^{\\frac{k+1}{2}} \\right) & \\mbox{, if } k \\mbox{ is odd} \\end{array} \\right.`
max         :math:`\\mathcal{A} (D^1, D^2, ..., D^k) = max_{ l} \\; D_{i,j}^l`
min         :math:`\\mathcal{A} (D^1, D^2, ..., D^k) = min_{ l} \\; D_{i,j}^l`
=========== ===========
"""

import numpy as np
from abc import ABC, abstractmethod


class Aggregator(ABC):
    """
    An abstract base class for *Aggregators*.
    If custom *Aggregators* are created,
    it is enough to derive from this class
    and use it whenever an *Aggregator* is needed.
    """

    @abstractmethod
    def aggregate(self, matrices):
        """
        The abstract method that is implemented by the concrete *Aggregators*.

        :param matrices: a list of similarity or dissimilarity matrices as 2D numpy arrays.
        :return: a single 2D numpy array.
        """
        pass


class AggregatorFactory:
    """
    The factory class for creating concrete instances of the implemented *Aggregators* with default values.
    """

    @staticmethod
    def create(aggregator):
        """
        Creates an instance of the given *Aggregator* name.

        :param aggregator: The name of the *Aggregator*, which can be ``mean``, ``median``, ``max`` or ``min``.

        :return: An instance of the *Aggregator*.

        :raise ValueError: The given *Aggregator* does not exist.
        """
        if aggregator == "mean":
            return MeanAggregator()
        elif aggregator == "median":
            return MedianAggregator()
        elif aggregator == "max":
            return MaxAggregator()
        elif aggregator == "min":
            return MinAggregator()
        else:
            raise ValueError(f"An aggregator of type {aggregator} does not exist.")


class MeanAggregator(Aggregator):
    """
    This class aggregates similarity or dissimilarity matrices using the ``mean``.
    Given :math:`k` similarity or dissimilarity matrices :math:`D^i \\in \\mathbb{R}^{n \\times n}`,
    the *MeanAggregator* calculates

    .. centered::
        :math:`\\mathcal{A} (D^1, D^2, ..., D^k) = \\frac{1}{k} \\sum_{i=1}^{k} D^i`.
    """

    def aggregate(self, matrices):
        """
        Calculates the mean of all given matrices along the zero axis.

        :param matrices: A list of 2D numpy arrays.
        :return: A 2D numpy array.
        """
        return np.mean(matrices, axis=0)


class MedianAggregator(Aggregator):
    """
    This class aggregates similarity or dissimilarity matrices using the ``median``.
    Given :math:`k` similarity or dissimilarity matrices :math:`D^i \\in \\mathbb{R}^{n \\times n}`,
    the *MedianAggregator* calculates

    .. centered::
        :math:`\\mathcal{A} (D^1, D^2, ..., D^k) = \\left{ \\begin{array}{ll} D^{\\frac{k}{2}}  & \\mbox{, if } k \\mbox{ is even} \\\\ \\frac{1}{2} \\left( D^{\\frac{k-1}{2}} + D^{\\frac{k+1}{2}} \\right) & \\mbox{, if } k \\mbox{ is odd} \\end{array} \\right.`
    """

    def aggregate(self, matrices):
        """
        Calculates the median of all given matrices along the zero axis.

        :param matrices: A list of 2D numpy arrays.
        :return: A 2D numpy array.
        """
        return np.median(matrices, axis=0)


class MaxAggregator(Aggregator):
    """
    This class aggregates similarity or dissimilarity matrices using the ``max``.
    Given :math:`k` similarity or dissimilarity matrices :math:`D^i \\in \\mathbb{R}^{n \\times n}`,
    the *MaxAggregator* calculates

    .. centered::
        :math:`\\mathcal{A} (D^1, D^2, ..., D^k) = max_{ l} \\; D_{i,j}^l`.
    """

    def aggregate(self, matrices):
        """
        Calculates the max of all given matrices along the zero axis.

        :param matrices: A list of 2D numpy arrays.
        :return: A 2D numpy array.
        """
        return np.max(matrices, axis=0)


class MinAggregator(Aggregator):
    """
    This class aggregates similarity or dissimilarity matrices using the ``min``.
    Given :math:`k` similarity or dissimilarity matrices :math:`D^i \\in \\mathbb{R}^{n \\times n}`,
    the *MinAggregator* calculates

    .. centered::
        :math:`\\mathcal{A} (D^1, D^2, ..., D^k) = min_{ l} \\; D_{i,j}^l`.
    """

    def aggregate(self, matrices):
        """
        Calculates the min of all given matrices along the zero axis.

        :param matrices: A list of 2D numpy arrays.
        :return: A 2D numpy array.
        """
        return np.min(matrices, axis=0)
