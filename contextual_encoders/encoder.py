"""
ContextualEncoder
====================================
The *ContextualEncoder* is the actual interface for using the Contextual Encoders library.
It is used to perform the contextual encoding of a given dataset.
Moreover, it inherits from the scikit-
learn `BaseEstimator <https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html>`_
and `TransformerMixin <https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html>`_
types and thus enable being used in scikit-
learn `Pipelines <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_.

Having a dataset :math:`X \\subset \\mathcal{F}`, with :math:`\\mathcal{F}` denoting the feature space,
the *ContextualEncoder* can be seen as
a map :math:`\\mathcal{E} : X \\subset \\mathcal{F} \\rightarrow  \\tilde{X} \\subset \\mathbb{R}^{m}`,
with :math:`m \\in \\mathbb{N}` being the (configurable) dimension of the encoding
and :math:`\\tilde{X}` the encoded dataset as vectors.

In other words, let :math:`n \\in \\mathbb{N}` be the amount of features.
The *ContextualEncoder* then takes :math:`n` features that are either numerical, categorical
or a mix of both and produces :math:`n` vectors of dimension :math:`m \\in \\mathbb{N}`.

.. note::

    Additionally, a similarity matrix :math:`S \\in \\mathbb{R}^{n \\times n}`
    and dissimilarity matrix :math:`D \\in \\mathbb{R}^{n \\times n}` will be calculated.

.. note::

    Assuming we have a dataset with :math:`n`rows and :math:`k` columns.
    Each column is called an attribute and each row is called a feature.
    One attribute of a particular feature can consist of multiple values.
    Those values are called the forms of the attribute.
    The forms can be separated e.g. with a comma.
    The contextual encoding then consists of the following steps:

    - Calculate a comparison value for each form of each attribute and feature using a :class:`.Measure`.
    - Combine the form comparison values to an attribute comparison value using a :class:`.Gatherer`.
    - Combine the attribute comparison values to a feature comparison value using an :class:`.Aggregator`.
    - Use an :class:`.Inverter` to either get a similarity value from a dissimilarity value or visa verse.
    - Collect all the feature comparison values and construct the similarity and dissimilarity
      matrix within the :class:`.MatrixComputer`.
    - Convert the similarity or dissimilarity matrix to a set of vectors using a :class:`.Reducer`.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from .measure import Measure, SimilarityMeasure, DissimilarityMeasure
from .aggregator import AggregatorFactory, Aggregator
from .computer import MatrixComputer
from .gatherer import Gatherer, GathererFactory
from .inverter import Inverter, InverterFactory
from .reducer import ReducerFactory, SimilarityMatrixReducer, Reducer
from .data_utils import DataUtils


class ContextualEncoder(BaseEstimator, TransformerMixin):
    """
    The interface for encoding contextual variables.
    """

    def __init__(
        self,
        measures,
        separator_token=",",
        gatherers="smm",
        aggregator="mean",
        inverters="sqrt",
        reducer="mds",
    ):
        """
        Initializes the *ContextualEncoder*.

        .. note::

            If no concrete instances but only names are specified for the components,
            an instance will be created with the default values.

        :param measures: A list of *Measures*.
            If :math:`k \\in \\mathbb{N}` columns should be encoded,
            the list needs to be of size :math:`k`.
            See :class:`.Measure` for currently implemented *Measures*
            and how custom *Measures* can be implemented.
        :param separator_token: A string for separating forms of attributes.
        :param gatherers: A list of either *Gatherer* instances or *Gatherer* names.
            If :math:`k \\in \\mathbb{N}` columns should be encoded,
            the list needs to be of size :math:`k`.
            If only one *Gatherer* should be used for all columns,
            a single object is enough and a list is not needed.
            See :class:`.Gatherer` for currently implemented *Gatherers*
            and how custom *Gatherers* can be implemented.
            See :class:`.GathererFactory` for the names of the implemented *Gatherers*.
        :param aggregator: Either an *Aggregator* instance or an *Aggregator* name.
            See :class:`.Aggregator` for currently implemented *Aggregators*
            and how custom *Aggregators* can be implemented.
            See :class:`.AggregatorFactory` for the names of the implemented *Aggregators*.
        :param inverters: A list of either *Inverter* instances or *Inverter* names.
            If :math:`k \\in \\mathbb{N}` columns should be encoded,
            the list needs to be of size :math:`k`.
            If only one *Inverter* should be used for all columns,
            a single object is enough and a list is not needed.
            See :class:`.Inverter` for currently implemented *Inverters*
            and how custom *Inverters* can be implemented.
            See :class:`.InverterFactory` for the names of the implemented *Inverters*.
        :param reducer: Either a *Reducer* instance or a *Reducer* name.
            See :class:`.Reducer` for currently implemented *Reducers*
            and how custom *Reducers* can be implemented.
            See :class:`.ReducerFactory` for the names of the implemented *Reducers*.
        """

        if isinstance(measures, Measure):
            self.__measures = [measures]
        else:
            self.__measures = measures

        self.__separator_token = separator_token

        if isinstance(gatherers, Gatherer) or isinstance(gatherers, str):
            gatherers = [gatherers]
        else:
            gatherers = gatherers

        self.__gatherers = []
        for i in range(0, len(gatherers)):
            if isinstance(gatherers[i], str):
                self.__gatherers.append(GathererFactory.create(gatherers[i]))
            elif isinstance(gatherers[i], Gatherer):
                self.__gatherers.append((gatherers[i]))
            else:
                raise ValueError(
                    "The specified Gatherer is either not of type Gatherer or could not be found"
                )

        if isinstance(aggregator, str):
            self.__aggregator = AggregatorFactory.create(aggregator)
        elif isinstance(aggregator, Aggregator):
            self.__aggregator = aggregator
        else:
            raise ValueError(
                "The specified Aggregator is either not of type Aggregator or could not be found"
            )

        if isinstance(inverters, Inverter) or isinstance(inverters, str):
            inverters = [inverters]
        else:
            inverters = inverters

        self.__inverters = []
        for i in range(0, len(inverters)):
            if isinstance(inverters[i], str):
                self.__inverters.append(InverterFactory.create(inverters[i]))
            elif isinstance(inverters[i], Inverter):
                self.__inverters.append(inverters[i])
            else:
                raise ValueError(
                    "The specified Inverter is either not of type Inverter or could not be found"
                )

        if isinstance(reducer, str):
            self.__reducer = ReducerFactory.create(reducer)
        elif isinstance(reducer, Reducer):
            self.__reducer = reducer
        else:
            raise ValueError(
                "The specified Reducer is either not of type Reducer or could not be found"
            )

        self.__computer = []
        for i in range(0, len(self.__measures)):
            self.__computer.append(
                MatrixComputer(
                    self.__measures[i],
                    self.__gatherers[i],
                    separator_token=self.__separator_token,
                )
            )

        self.__similarity_matrix = None
        self.__dissimilarity_matrix = None

        return

    def transform(self, x):
        """
        Encodes the given contextual variables.

        :param x: The data as numpy array, pandas dataframe or python list format.
        :return: The encoded data as numpy array.
        """

        similarity_matrices = []
        dissimilarity_matrices = []

        x_df = DataUtils.ensure_pandas_dataframe(x)

        for col in x_df.columns:

            matrix = self.__computer[col].compute(x_df[col])

            if isinstance(self.__measures[col], SimilarityMeasure):
                similarity_matrices.append(matrix)
                dissimilarity_matrices.append(
                    self.__inverters[col].similarity_to_dissimilarity(matrix)
                )
            elif isinstance(self.__measures[col], DissimilarityMeasure):
                dissimilarity_matrices.append(matrix)
                similarity_matrices.append(
                    self.__inverters[col].dissimilarity_to_similarity(matrix)
                )

        aggregated_similarity_matrix = self.__aggregator.aggregate(similarity_matrices)
        aggregated_dissimilarity_matrix = self.__aggregator.aggregate(
            dissimilarity_matrices
        )

        self.__similarity_matrix = aggregated_similarity_matrix
        self.__dissimilarity_matrix = aggregated_dissimilarity_matrix

        if isinstance(self.__reducer, SimilarityMatrixReducer):
            data_points = self.__reducer.reduce(self.__similarity_matrix)
        else:
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
