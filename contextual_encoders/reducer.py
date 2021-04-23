from abc import ABC, abstractmethod
from enum import Enum
from sklearn.manifold import MDS


class Reducer(ABC):

    def __init__(self, n_components):
        self.__n_components = n_components

    @abstractmethod
    def reduce(self, matrix):
        pass


class ReducerType(Enum):
    MultidimensionalScaling = 'mds'

    @staticmethod
    def create(reducer_type, **kwargs):
        if 'n_components' not in kwargs:
            kwargs['n_components'] = 2
        if 'metric' not in kwargs:
            kwargs['metric'] = True

        if reducer_type == ReducerType.MultidimensionalScaling:
            return MultidimensionalScalingReducer(n_components=kwargs['n_components'], metric=kwargs['metric'])
        else:
            raise ValueError(f'A reducer of type {reducer_type} does not exist.')


class MultidimensionalScalingReducer(Reducer):

    def __init__(self, n_components, metric):
        super().__init__(n_components)
        self.__mds = MDS(n_components, metric=metric, dissimilarity='precomputed')

    def reduce(self, matrix):
        return self.__mds.fit_transform(matrix)
