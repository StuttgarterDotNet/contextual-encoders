"""
Inverter
====================================
An *Inverter* is used to calculate a dissimilarity value given a similarity value and vice versa.
It can be seen as a one-to-one mapping :math:`\\mathcal{I} : [0,1] \\rightarrow [0,1]`.


Currently, the following *Inverters* are implement:

=========== ===========
Name        Formula
----------- -----------
lin         :math:`\\mathcal{I} (s) = 1 - s`
sqrt        :math:`\\mathcal{I} (s) = \\sqrt{1 - s}`
exp         :math:`\\mathcal{I} (s) = 2 - e^{ln(2) \\cdot s}`
cos         :math:`\\mathcal{I} (s) = cos(\\frac{\\pi}{2} \\cdot s)`
=========== ===========

.. note::

    If a custom inverter is implemented, make sure that the function is invertible and
    the definition range and value range is :math:`[0, 1]`.
"""

import numpy as np
from abc import ABC, abstractmethod


class Inverter(ABC):
    """
    An abstract base class for all concrete *Inverter* implementations.
    """

    @abstractmethod
    def similarity_to_dissimilarity(self, similarity_matrix):
        """
        Calculates a dissimilarity matrix given a similarity matrix.
        :param similarity_matrix: a similarity matrix as 2D numpy array.
        :return: a dissimilarity matrix as 2D numpy array.
        """
        pass

    @abstractmethod
    def dissimilarity_to_similarity(self, dissimilarity_matrix):
        """
        Calculates a similarity matrix given a dissimilarity matrix.
        :param dissimilarity_matrix: a dissimilarity matrix as 2D numpy array.
        :return: a similarity matrix as 2D numpy array.
        """
        pass


class InverterFactory:
    """
    A factory class to create concrete *Inverter* instances with default values.
    """

    @staticmethod
    def create(inverter):
        """
        Creates an instance of the given *Inverter* name.

        :param inverter: The name of the *Inverter*, which can be ``lin``, ``sqrt``, ``exp`` or ``cos``.
        :return: An instance of the *Inverter*.
        :raise ValueError: The given *Inverter* does not exist.
        """
        if inverter == "lin":
            return LinearInverter()
        elif inverter == "sqrt":
            return SqrtInverter()
        elif inverter == "exp":
            return ExponentialInverter()
        elif inverter == "cos":
            return CosineInverter()
        else:
            raise ValueError(f"An inverter of type {inverter} does not exist.")


class LinearInverter(Inverter):
    """
    An *Inverter* that converts a similarity matrix to a dissimilarity matrix and vice versa using a linear ansatz.
    It can be used as the ``lin`` option.
    """

    def similarity_to_dissimilarity(self, similarity_matrix):
        """
        Converts the given similarity matrix to a dissimilarity matrix accordingly
        to :math:`\\mathcal{I} (s) = 1 - s`, with :math:`s` being the similarity matrix.
        The operations are considered as elementwise.

        :param similarity_matrix: A similarity matrix as 2D numpy array.
        :return: A dissimilarity matrix as 2D numpy array.
        """
        return 1.0 - similarity_matrix

    def dissimilarity_to_similarity(self, dissimilarity_matrix):
        """
        Converts the given dissimilarity matrix to a similarity matrix accordingly
        to :math:`\\mathcal{I}^{-1} (d) = 1 - d`, with :math:`d` being the dissimilarity matrix.
        The operations are considered as elementwise.

        :param dissimilarity_matrix: A dissimilarity matrix as 2D numpy array.
        :return: A similarity matrix as 2D numpy array.
        """
        return 1.0 - dissimilarity_matrix


class SqrtInverter(Inverter):
    """
    An *Inverter* that converts a similarity matrix to a dissimilarity matrix and vice versa using a sqrt ansatz.
    It can be used as the ``sqrt`` option.
    """

    def similarity_to_dissimilarity(self, similarity_matrix):
        """
        Converts the given similarity matrix to a dissimilarity matrix accordingly
        to :math:`\\mathcal{I} (s) = \\sqrt{1 - s}`, with :math:`s` being the similarity matrix.
        The operations are considered as elementwise.

        :param similarity_matrix: A similarity matrix as 2D numpy array.
        :return: A dissimilarity matrix as 2D numpy array.
        """
        return np.sqrt(1.0 - similarity_matrix)

    def dissimilarity_to_similarity(self, dissimilarity_matrix):
        """
        Converts the given dissimilarity matrix to a similarity matrix accordingly
        to :math:`\\mathcal{I}^{-1} (d) = 1 - d^2`, with :math:`d` being the dissimilarity matrix.
        The operations are considered as elementwise.

        :param dissimilarity_matrix: A dissimilarity matrix as 2D numpy array.
        :return: A similarity matrix as 2D numpy array.
        """
        return 1.0 - np.power(dissimilarity_matrix, 2)


class ExponentialInverter(Inverter):
    """
    An *Inverter* that converts a similarity matrix to a dissimilarity matrix
    and vice versa using an exponential ansatz.
    It can be used as the ``exp`` option.
    """

    def similarity_to_dissimilarity(self, similarity_matrix):
        """
        Converts the given similarity matrix to a dissimilarity matrix accordingly
        to :math:`\\mathcal{I} (s) = 2 - e^{ln(2) \\cdot s}`, with :math:`s` being the similarity matrix.
        The operations are considered as elementwise.

        :param similarity_matrix: A similarity matrix as 2D numpy array.
        :return: A dissimilarity matrix as 2D numpy array.
        """
        return 2.0 - np.exp(np.log(2.0) * similarity_matrix)

    def dissimilarity_to_similarity(self, dissimilarity_matrix):
        """
        Converts the given dissimilarity matrix to a similarity matrix accordingly
        to :math:`\\mathcal{I}^{-1} (d) = \\frac{1}{ln(2)} ln(2 - d)`, with :math:`d` being the dissimilarity matrix.
        The operations are considered as elementwise.

        :param dissimilarity_matrix: A dissimilarity matrix as 2D numpy array.
        :return: A similarity matrix as 2D numpy array.
        """
        return np.log(2.0 - dissimilarity_matrix) / np.log(2.0)


class CosineInverter(Inverter):
    """
    An *Inverter* that converts a similarity matrix to a dissimilarity matrix
    and vice versa using a cosine ansatz.
    It can be used as the ``cos`` option.
    """

    def similarity_to_dissimilarity(self, similarity_matrix):
        """
        Converts the given similarity matrix to a dissimilarity matrix accordingly
        to :math:`\\mathcal{I} (s) = cos(\\frac{\\pi}{2} \\cdot s)`, with :math:`s` being the similarity matrix.
        The operations are considered as elementwise.

        :param similarity_matrix: A similarity matrix as 2D numpy array.
        :return: A dissimilarity matrix as 2D numpy array.
        """
        return np.cos((np.pi * similarity_matrix) / 2.0)

    def dissimilarity_to_similarity(self, dissimilarity_matrix):
        """
        Converts the given dissimilarity matrix to a similarity matrix accordingly
        to :math:`\\mathcal{I}^{-1} (d) = \\frac{2}{\\pi} acos(d)`, with :math:`d` being the dissimilarity matrix.
        The operations are considered as elementwise.

        :param dissimilarity_matrix: A dissimilarity matrix as 2D numpy array.
        :return: A similarity matrix as 2D numpy array.
        """
        return (2.0 / np.pi) * np.arccos(dissimilarity_matrix)
