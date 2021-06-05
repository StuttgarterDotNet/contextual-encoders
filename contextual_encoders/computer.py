"""
Computer
====================================

"""

import numpy as np
from .gatherer import GathererType, Identity


class MatrixComputer:
    def __init__(self, measure, gatherer, separator_token):
        self.__measure = measure
        self.__separator_token = separator_token

        if self.__measure.can_handle_multiple_values():
            self.__gatherer = GathererType.create(Identity)
        else:
            self.__gatherer = GathererType.create(gatherer)

    def compute(self, data):
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
