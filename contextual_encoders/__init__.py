"""
.. module:: contextual_encoders
  :synopsis:
  :platform:
"""

from contextual_encoders.aggregator import (
    Aggregator,
    AggregatorType,
    MeanAggregator,
    MedianAggregator,
    MaxAggregator,
    MinAggregator,
)
from contextual_encoders.comparer import Comparer, WuPalmerComparer
from contextual_encoders.google_comparer import GoogleComparer
from contextual_encoders.computer import SimilarityMatrixComputer
from contextual_encoders.context import TreeContext
from contextual_encoders.encoder import ContextualEncoder
from contextual_encoders.gatherer import (
    Gatherer,
    GathererType,
    IdentityGatherer,
    FirstValueGatherer,
    SymMaxMeanGatherer,
)
from contextual_encoders.inverter import (
    Inverter,
    InverterType,
    IdentityInverter,
    LinearInverter,
    SqrtInverter,
    ExponentialInverter,
    GaussInverter,
    HyperbolicInverter,
    CosineInverter,
)
from contextual_encoders.reducer import (
    Reducer,
    ReducerType,
    MultidimensionalScalingReducer,
)
