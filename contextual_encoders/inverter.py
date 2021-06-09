"""
Inverter
====================================
A *Gatherer* is used to combine a set of pairwise attribute measures to a single measure.

.. note::

    If a measure can handle multiple values, a Gatherer is not needed.
"""

import numpy as np
from abc import ABC, abstractmethod

Linear = "lin"
Sqrt = "sqrt"
Exponential = "exp"
Cosine = "cos"


class Inverter(ABC):
    @abstractmethod
    def similarity_to_dissimilarity(self, similarity_matrix):
        pass

    @abstractmethod
    def dissimilarity_to_similarity(self, dissimilarity_matrix):
        pass


class InverterType:
    @staticmethod
    def create(inverter_type):
        if inverter_type == Linear:
            return LinearInverter()
        elif inverter_type == Sqrt:
            return SqrtInverter()
        elif inverter_type == Exponential:
            return ExponentialInverter()
        elif inverter_type == Cosine:
            return CosineInverter()
        else:
            raise ValueError(f"An inverter of type {inverter_type} does not exist.")


class LinearInverter(Inverter):
    def similarity_to_dissimilarity(self, similarity_matrix):
        return 1.0 - similarity_matrix

    def dissimilarity_to_similarity(self, dissimilarity_matrix):
        return 1.0 - dissimilarity_matrix


class SqrtInverter(Inverter):
    def similarity_to_dissimilarity(self, similarity_matrix):
        return np.sqrt(1.0 - similarity_matrix)

    def dissimilarity_to_similarity(self, dissimilarity_matrix):
        return 1.0 - np.power(dissimilarity_matrix, 2)


class ExponentialInverter(Inverter):
    def similarity_to_dissimilarity(self, similarity_matrix):
        return 2.0 - np.exp(np.log(2.0) * similarity_matrix)

    def dissimilarity_to_similarity(self, dissimilarity_matrix):
        return np.log(2.0 - dissimilarity_matrix) / np.log(2.0)


class CosineInverter(Inverter):
    def similarity_to_dissimilarity(self, similarity_matrix):
        return np.cos((np.pi * similarity_matrix) / 2.0)

    def dissimilarity_to_similarity(self, dissimilarity_matrix):
        return (2.0 / np.pi) * np.arccos(dissimilarity_matrix)
