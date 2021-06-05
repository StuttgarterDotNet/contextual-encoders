from abc import ABC, abstractmethod

Identity = "id"
First = "first"
SymMaxMean = "smm"


class Gatherer(ABC):
    def __init__(self):
        self._measure = None

    @abstractmethod
    def _gather(self, first, second):
        pass

    def set_measure(self, comparer):
        self._measure = comparer

    def gather(self, first, second):
        if self._measure is None:
            raise ValueError("No measure is specified")

        # if we don't have lists, convert them to list
        if not isinstance(first, list):
            first = [first]
        if not isinstance(second, list):
            second = [second]

        return self._gather(first, second)


class GathererType:
    @staticmethod
    def create(gatherer_type):
        if gatherer_type == Identity:
            return IdentityGatherer()
        elif gatherer_type == First:
            return FirstValueGatherer()
        elif gatherer_type == SymMaxMean:
            return SymMaxMeanGatherer()
        else:
            raise ValueError(f"A gatherer of type {gatherer_type} does not exist.")


class IdentityGatherer(Gatherer):
    def _gather(self, first, second):
        return self._measure.compare(first, second)


class FirstValueGatherer(Gatherer):
    def _gather(self, first, second):
        first = first[0]
        second = second[0]

        return self._measure.compare(first, second)


class SymMaxMeanGatherer(Gatherer):
    def _gather(self, first, second):
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
