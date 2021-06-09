"""
Gatherer
====================================
A *Gatherer* is used to combine a set of pairwise attribute measures to a single measure.

.. note::

    If a measure can handle multiple values, a Gatherer is not needed.
"""

from abc import ABC, abstractmethod

Identity = "id"
First = "first"
SymMaxMean = "smm"


class Gatherer(ABC):
    """
    The abstract base class of all Gatherer.
    """
    def __init__(self):
        """
        Initializes the Gatherer.
        """
        self._measure = None

    @abstractmethod
    def _gather(self, first, second):
        """
        The abstract method to gather two attributes.
        This class needs to be implemented by concrete instances of Gatherer.

        .. note::

            Multiple values, e.g. comma separated, are exclusively possible.

        :param first: The value of the first attribute.
        :param second: The value of the second attribute.
        :return: The aggregated value.
        """
        pass

    def set_measure(self, measure):
        """
        Sets the measure for the Gatherer.
        
        :param measure: The measure.
        """
        self._measure = measure

    def gather(self, first, second):
        """
        Combines the given attributes.

        .. note::

            Multiple values, e.g. comma separated, are exclusively possible.

        :param first: The value of the first attribute.
        :param second: The value of the second attribute.
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
    The factory class for creating Gatherer.
    """
    @staticmethod
    def create(gatherer_type):
        """
        Creates a Gatherer given the type.

        :param gatherer_type: The type of the Gatherer which can be ``id``, ``first`` or ``smm``.
        :return: The concrete instance of the Gatherer.
        """
        if gatherer_type == Identity:
            return IdentityGatherer()
        elif gatherer_type == First:
            return FirstValueGatherer()
        elif gatherer_type == SymMaxMean:
            return SymMaxMeanGatherer()
        else:
            raise ValueError(f"A gatherer of type {gatherer_type} does not exist.")


class IdentityGatherer(Gatherer):
    """
    A Gatherer that let the measure decide how to handle multiple values.
    """
    def _gather(self, first, second):
        """
        Calling the measure without handling multiple values at Gatherer level.

        :param first: The value of the first attribute.
        :param second: The value of the second attribute.
        :return: The value returned from the measure.
        """
        return self._measure.compare(first, second)


class FirstValueGatherer(Gatherer):
    """
    A Gatherer only measuring the first values of the attributes.
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
    A symmetrical maximum gatherer implementation.
    """
    def _gather(self, first, second):
        """
        Gathers two attributes in a symmetrical manner.

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
