"""
Gatherer
====================================
A *Gatherer* is used to combine the form comparison values to an attribute comparison value.
I.e. when an attribute of a feature contains multiple values (forms of an attribute),
the *Gatherer* will combine the pairwise form comparison values to a single attribute comparison value.

Let :math:`x, y \\in F` be two features from the feature space :math:`F`.
Each feature consists of :math:`k` attributes and each attribute can have up to :math:`l` forms.
A form of an attribute of the feature :math:`x` can then be denoted as :math:`x_{a,i}`,
with :math:`a` being the attribute and :math:`i` being the form.
For simplicity, we just denote it as :math:`\\tilde{x}_i` and skip the attribute index.
A *Measure* is then defined as :math:`\\mathcal{M} : (\\tilde{x}_i, \\tilde{y}_j) \\rightarrow [0,1]`, i.e.
it maps a similarity or dissimilarity value to each attribute form.

The *Gatherer* uses the *Measure* together with all the attribute forms and calculates a single attribute
comparison value. Hence, it can be seen as a
mapping :math:`\\mathcal{G} : (x, y, \\mathcal{M}) \\mapsto g \\in \\mathbb{R}`.


Currently, the following *Gatherers* are implement:

=========== ===========
Name        Formula
----------- -----------
id          :math:`\\mathcal{G} (x, y, \\mathcal{M}) = \\mathcal{M}(x, y)`
first       :math:`\\mathcal{G} (x, y, \\mathcal{M}) = \\mathcal{M}(x_1, y_1)`
smm         :math:`\\mathcal{G} (x, y, \\mathcal{M}) = \\frac{1}{2} \\Big( \\frac{1}{|x|} \\sum_{i=1}^{l_x} \\mathcal{M}(x_i, \\tilde{y}) + \\frac{1}{|y|} \\sum_{i=1}^{l_y} \\mathcal{M}(\\tilde{x}, y_i) \\Big)`
=========== ===========

.. note::

    If a *Measure* has the property ``multiple_values``,
    it accepts all forms of an attribute as input and can calculate an attribute comparison value,
    rather then an attribute form comparison value.

    In this case, a *Gatherer* is not needed.
"""

from abc import ABC, abstractmethod


class Gatherer(ABC):
    """
    The abstract base class of all *Gatherers*.
    """

    def __init__(self):
        """
        Initializes the *Gatherer*.
        """
        self._measure = None

    @abstractmethod
    def _gather(self, first, second):
        """
        The abstract method to gather all attribute form comparison values of two features.
        This class needs to be implemented by concrete instances of *Gatherers*.

        .. note::

            Multiple values, e.g. comma separated, are exclusively possible.

        :param first: A list of the value(s) of the first attribute.
        :param second: A list of the value(s) of the second attribute.
        :return: The aggregated value.
        """
        pass

    def set_measure(self, measure):
        """
        Sets the *Measure* for the *Gatherer*.

        :param measure: The *Similarity* or *Dissimilarity Measure*,
            see :class:`.SimilarityMeasure` and :class:`DissimilarityMeasure`.
        """
        self._measure = measure

    def gather(self, first, second):
        """
        Combines the given attributes.

        .. note::

            Multiple values, e.g. comma separated, are exclusively possible.

        :param first: A list of the value(s) of the first attribute.
        :param second: A list of the value(s) of the second attribute.
        :return: The aggregated value.
        """
        if self._measure is None:
            raise ValueError("No measure is specified")

        # if we don't have lists, convert them to list
        if not isinstance(first, list):
            first = [first]
        if not isinstance(second, list):
            second = [second]

        return self._gather(first, second)


class GathererFactory:
    """
    The factory class for creating *Gatherers*.
    """

    @staticmethod
    def create(gatherer):
        """
        Creates a *Gatherer* given the name.

        :param gatherer: The name of the *Gatherer*, which can be ``id``, ``first`` or ``smm``.
        :return: The concrete instance of the *Gatherer*.
        """
        if gatherer == "id":
            return IdentityGatherer()
        elif gatherer == "first":
            return FirstValueGatherer()
        elif gatherer == "smm":
            return SymMaxMeanGatherer()
        else:
            raise ValueError(f"A gatherer of type {gatherer} does not exist.")


class IdentityGatherer(Gatherer):
    """
    A *Gatherer* that let the *Measure* decide how to handle multiple attribute forms.
    It can be seen as a mapping

    .. centered::
        :math:`\\mathcal{G} (x, y, \\mathcal{M}) = \\mathcal{M}(x, y)`,

    with :math:`\\mathcal{M}` being the *Similarity* or *Dissimilarity Measure*
    and :math:`x, y` being the same attributes from different features.
    """

    def _gather(self, first, second):
        """
        Calling the *Measure* without handling multiple values at *Gatherer* level.

        :param first: The value of the first attribute.
        :param second: The value of the second attribute.
        :return: The value returned from the measure.
        """
        return self._measure.compare(first, second)


class FirstValueGatherer(Gatherer):
    """
    A *Gatherer* that let only uses the first forms of the attributes.
    It can be seen as a mapping

    .. centered::
        :math:`\\mathcal{G} (x, y, \\mathcal{M}) = \\mathcal{M}(x_1, y_1)`,

    with :math:`\\mathcal{M}` being the *Similarity* or *Dissimilarity Measure*
    and :math:`x, y` being the same attributes from different features.
    """

    def _gather(self, first, second):
        """
        Gather the given attributes with only measuring their first values.

        :param first: The value of the first attribute.
        :param second: The value of the second attribute.
        :return: The combined value.
        """
        first = first[0]
        second = second[0]

        return self._measure.compare(first, second)


class SymMaxMeanGatherer(Gatherer):
    """
    A *Gatherer* that symmetrically measures all pairwise attribute forms but only uses the maximum value.
    It can be seen as a mapping

    .. centered::
        :math:`\\mathcal{G} (x, y, \\mathcal{M}) = \\frac{1}{2} \\Big( \\frac{1}{|x|} \\sum_{i=1}^{l_x} \\mathcal{M}(x_i, \\tilde{y_i}) + \\frac{1}{|y|} \\sum_{i=1}^{l_y} \\mathcal{M}(\\tilde{x_i}, y_i) \\Big)`

    with :math:`\\mathcal{M}` being the *Similarity* or *Dissimilarity Measure*, :math:`x, y`
    being the same attributes from different features, :math:`|x|` the amount of attribute
    forms of the attribute :math:`x` and :math:`\\tilde{x_i} = argmax_{j=1,...,l_x} \\mathcal{M}(x_j, y_i)`.
    """

    def _gather(self, first, second):
        """
        Gathers two attributes symmetrically based on the maximum comparison value.

        :param first: The value of the first attribute.
        :param second: The value of the second attribute.
        :return: The combined value.
        """
        sum1 = 0.0
        sum2 = 0.0

        # sum over a in first
        for a in first:
            # get max value_compare(a, b) with b in second
            max_value = 0.0
            for b in second:
                temp = self._measure.compare(a, b)
                if temp > max_value:
                    max_value = temp
            sum1 += max_value

        # normalize to size of first
        sum1 /= len(first)

        # sum over b in second
        for b in second:
            # get max value_compare(b, a) with a in first
            max_value = 0.0
            for a in first:
                temp = self._measure.compare(b, a)
                if temp > max_value:
                    max_value = temp
            sum2 += max_value

        # normalize to size of second
        sum2 /= len(second)

        # combine both sums
        return 0.5 * (sum1 + sum2)
