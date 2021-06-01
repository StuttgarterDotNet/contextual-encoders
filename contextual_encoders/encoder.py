from sklearn.base import BaseEstimator, TransformerMixin
from .comparer import Comparer
from .aggregator import AggregatorType, Mean
from .computer import SimilarityMatrixComputer
from .gatherer import SymMaxMean
from .inverter import InverterType, Linear
from .reducer import ReducerType, MultidimensionalScaling
from .data_utils import DataUtils


class ContextualEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        comparer,
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
        self.__aggregator = AggregatorType.create(aggregator)
        self.__inverter = InverterType.create(inverter, **kwargs)
        self.__reducer = ReducerType.create(reducer, **kwargs)
        self.__matrix = None

        if isinstance(comparer, Comparer):
            comparer = [comparer]

        for i in range(0, len(comparer)):
            self.__computer.append(
                SimilarityMatrixComputer(
                    comparer[i], gatherer, kwargs["separator_token"]
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
        matrices = []
        x_df = DataUtils.ensure_pandas_dataframe(x)
        self.__cols = self.infer_columns(x_df)

        for col in self.__cols:
            matrix = self.__computer[col].compute(x_df[col])
            inverted_matrix = self.__inverter.invert(matrix)
            matrices.append(inverted_matrix)

        aggregated_matrix = self.__aggregator.aggregate(matrices)
        self.__matrix = aggregated_matrix

        data_points = self.__reducer.reduce(aggregated_matrix)

        return data_points

    def get_matrix(self):
        return self.__matrix
