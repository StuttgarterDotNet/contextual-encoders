"""
Reducer
====================================
A *Gatherer* is used to combine a set of pairwise attribute measures to a single measure.

.. note::

    If a measure can handle multiple values, a Gatherer is not needed.
"""

from abc import ABC, abstractmethod
from sklearn.manifold import MDS

MultidimensionalScaling = "mds"


class Reducer(ABC):
    def __init__(self, n_components):
        self.__n_components = n_components

    @abstractmethod
    def reduce(self, matrix):
        pass


class ReducerType:
    @staticmethod
    def create(reducer_type, **kwargs):
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


class MultidimensionalScalingReducer(Reducer):
    def __init__(self, n_components, metric):
        super().__init__(n_components)
        self.__mds = MDS(n_components, metric=metric, dissimilarity="precomputed")

    def reduce(self, matrix):
        return self.__mds.fit_transform(matrix)

    def get_stress(self):
        return self.__mds.stress_
