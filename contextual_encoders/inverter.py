import math
from abc import ABC, abstractmethod
from enum import Enum


class Inverter(ABC):

    @abstractmethod
    def invert(self, similarity_matrix):
        pass


class InverterType(Enum):

    Identity = 'id'
    Linear = 'lin'
    Sqrt = 'sqrt'
    Exponential = 'exp'
    Gauss = 'gauss'
    Hyperbolic = 'hyp'
    Cosine = 'cos'

    @staticmethod
    def create(inverter_type, **kwargs):
        if 'max_similarity' not in kwargs:
            kwargs['max_similarity'] = 1.0
        if 'degree' not in kwargs:
            kwargs['degree'] = 1

        if inverter_type == InverterType.Identity:
            return IdentityInverter()
        elif inverter_type == InverterType.Linear:
            return LinearInverter(max_similarity=kwargs['max_similarity'])
        elif inverter_type == InverterType.Sqrt:
            return SqrtInverter(max_similarity=kwargs['max_similarity'])
        elif inverter_type == InverterType.Exponential:
            return ExponentialInverter(max_similarity=kwargs['max_similarity'])
        elif inverter_type == InverterType.Gauss:
            return GaussInverter(max_similarity=kwargs['max_similarity'])
        elif inverter_type == InverterType.Hyperbolic:
            return HyperbolicInverter(max_similarity=kwargs['max_similarity'], degree=kwargs['degree'])
        elif inverter_type == InverterType.Cosine:
            return CosineInverter(max_similarity=kwargs['max_similarity'])
        else:
            raise ValueError(f'An inverter of type {inverter_type} does not exist.')


class IdentityInverter(Inverter):

    def invert(self, similarity_matrix):
        return similarity_matrix


class LinearInverter(Inverter):

    def __init__(self, max_similarity):
        self.max_similarity = max_similarity

    def invert(self, similarity_matrix):
        return 1.0 - similarity_matrix / self.max_similarity


class SqrtInverter(Inverter):

    def __init__(self, max_similarity):
        self.max_similarity = max_similarity

    def invert(self, similarity_matrix):
        norm = math.sqrt(2.0) * self.max_similarity
        return (1.0 / norm) * math.sqrt(norm - 2.0 * similarity_matrix)


class ExponentialInverter(Inverter):

    def __init__(self, max_similarity):
        self.max_similarity = max_similarity

    def invert(self, similarity_matrix):
        return math.exp(-similarity_matrix / self.max_similarity)


class GaussInverter(Inverter):

    def __init__(self, max_similarity):
        self.max_similarity = max_similarity

    def invert(self, similarity_matrix):
        return math.exp(-math.pow(similarity_matrix / self.max_similarity, 2))


class HyperbolicInverter(Inverter):

    def __init__(self, max_similarity, degree):
        self.max_similarity = max_similarity
        self.degree = degree

    def invert(self, similarity_matrix):
        return 1.0 / (1.0 + pow(similarity_matrix / self.max_similarity, self.degree))


class CosineInverter(Inverter):

    def __init__(self, max_similarity):
        self.max_similarity = max_similarity

    def invert(self, similarity_matrix):
        return math.cos((math.pi * similarity_matrix)/(2.0 * self.max_similarity))
