import numpy as np
from abc import ABC, abstractmethod


class Aggregator(ABC):

    @abstractmethod
    def aggregate(self, matrices):
        pass


class AggregatorType:

    Mean = 'mean',
    Median = 'median'
    Max = 'max',
    Min = 'min'

    @staticmethod
    def create(aggregator_type):
        if aggregator_type == AggregatorType.Mean:
            return MeanAggregator()
        elif aggregator_type == AggregatorType.Median:
            return MedianAggregator()
        elif aggregator_type == AggregatorType.Max:
            return MaxAggregator()
        elif aggregator_type == AggregatorType.Min:
            return MinAggregator()
        else:
            raise ValueError(f'An aggregator of type {aggregator_type} does not exist.')


class MeanAggregator(Aggregator):

    def aggregate(self, matrices):
        return np.mean(matrices, axis=0)


class MedianAggregator(Aggregator):

    def aggregate(self, matrices):
        return np.median(matrices, axis=0)


class MaxAggregator(Aggregator):

    def aggregate(self, matrices):
        return np.max(matrices, axis=0)


class MinAggregator(Aggregator):

    def aggregate(self, matrices):
        return np.min(matrices, axis=0)
