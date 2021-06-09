"""
ContextualEncoder
====================================
The *ContextualEncoder* is the actual interface for using the Contextual Encoders library.
It is used to perform the contextual encoding of a given dataset.
Moreover, it inherits from the scikit-learn `BaseEstimator <https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html>`_
and `TransformerMixin <https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html>`_
types and thus enable being used in scikit-learn `Pipelines <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from .measure import Measure, SimilarityMeasure, DissimilarityMeasure
from .aggregator import AggregatorFactory, Mean
from .computer import MatrixComputer
from .gatherer import SymMaxMean
from .inverter import InverterType, Linear
from .reducer import ReducerType, MultidimensionalScaling
from .data_utils import DataUtils


class ContextualEncoder(BaseEstimator, TransformerMixin):
    # TODO: Remove the **kwargs keyword accordingly to
    #  https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
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
        """
        Initializes the *ContextualEncoder*.

        :param measures: A measure.
        :param cols: Pandas columns.
        :param inverter: The inverter
        :param gatherer: The gatherer.
        :param aggregator: The aggregator.
        :param reducer: The reducer.
        :param kwargs: Additional keywords.
        """
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
                MatrixComputer(measures[i], gatherer, kwargs["separator_token"])
            )

        return

    def __infer_columns(self, x):
        """
        Infers categorical columns form the given data.

        :param x: The data in pandas dataframe format.
        :return: A list of column names that are of categorical type.
        """
        if self.__cols is not None:
            return self.__cols
        elif len(x) == 0:
            self.__cols = []
            return []
        else:
            self.__cols = DataUtils.get_non_float_columns(x)

        return self.__cols

    def fit_transform(self, x, y=None, **fit_params):
        """
        Encodes the given data.

        :param x: The data as numpy array, pandas dataframe or python list format.
        :param y: TBA.
        :param fit_params: TBA.
        :return: The transformed data.
        """
        similarity_matrices = []
        dissimilarity_matrices = []

        x_df = DataUtils.ensure_pandas_dataframe(x)
        self.__cols = self.__infer_columns(x_df)

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
        """
        Gets the similarity matrix.

        :return: The similarity matrix as 2D numpy array.
        """
        return self.__similarity_matrix

    def get_dissimilarity_matrix(self):
        """
        Gets the dissimilarity matrix.

        :return: The dissimilarity matrix as 2D numpy array.
        """
        return self.__dissimilarity_matrix
