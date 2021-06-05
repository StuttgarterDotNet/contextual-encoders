from sklearn.base import BaseEstimator, TransformerMixin
from .measure import Measure, SimilarityMeasure, DissimilarityMeasure
from .aggregator import AggregatorFactory, Mean
from .computer import SimilarityMatrixComputer
from .gatherer import SymMaxMean
from .inverter import InverterType, Linear
from .reducer import ReducerType, MultidimensionalScaling
from .data_utils import DataUtils


class ContextualEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        measures,
        cols=None,
        inverter=Linear,
        gatherer=SymMaxMean,
        aggregator=Mean,
        reducer=MultidimensionalScaling,
        **kwargs
    ):
        if "separator_token" not in kwargs:
            kwargs["separator_token"] = ","

        self.__computer = []
        self.__cols = cols
        self.__aggregator = AggregatorFactory.create(aggregator)
        self.__inverter = InverterType.create(inverter)
        self.__reducer = ReducerType.create(reducer, **kwargs)
        self.__similarity_matrix = None
        self.__dissimilarity_matrix = None

        if isinstance(measures, Measure):
            measures = [measures]

        self.__measures = measures

        for i in range(0, len(self.__measures)):
            self.__computer.append(
                SimilarityMatrixComputer(
                    measures[i], gatherer, kwargs["separator_token"]
                )
            )

        return

    def infer_columns(self, x):
        if self.__cols is not None:
            return self.__cols
        elif len(x) == 0:
            self.__cols = []
            return []
        else:
            self.__cols = DataUtils.get_non_float_columns(x)

        return self.__cols

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        similarity_matrices = []
        dissimilarity_matrices = []

        x_df = DataUtils.ensure_pandas_dataframe(x)
        self.__cols = self.infer_columns(x_df)

        for col in self.__cols:
            matrix = self.__computer[col].compute(x_df[col])

            if isinstance(self.__measures[col], SimilarityMeasure):
                similarity_matrices.append(matrix)
                dissimilarity_matrices.append(
                    self.__inverter.similarity_to_dissimilarity(matrix)
                )
            elif isinstance(self.__measures[col], DissimilarityMeasure):
                dissimilarity_matrices.append(matrix)
                similarity_matrices.append(
                    self.__inverter.dissimilarity_to_similarity(matrix)
                )

        aggregated_similarity_matrix = self.__aggregator.aggregate(similarity_matrices)
        aggregated_dissimilarity_matrix = self.__aggregator.aggregate(
            dissimilarity_matrices
        )

        self.__similarity_matrix = aggregated_similarity_matrix
        self.__dissimilarity_matrix = aggregated_dissimilarity_matrix

        data_points = self.__reducer.reduce(self.__dissimilarity_matrix)

        return data_points

    def get_similarity_matrix(self):
        return self.__similarity_matrix

    def get_dissimilarity_matrix(self):
        return self.__dissimilarity_matrix
