import pandas as pd
import numpy as np


class DataUtils:
    """
    A helper class containing static functions for data handling.
    """

    @staticmethod
    def ensure_pandas_dataframe(x):
        """
        Ensure that the given data is of pandas dataframe.
        If not, the data will be converted.
        Additionally, the names of the columns will be replaced by indices.

        :param x: The data to check in either pandas dataframe, pandas series, numpy array or python list format.
        :return: The pandas dataframe representing the data.
        """
        if isinstance(x, pd.DataFrame):
            x_df = x.reindex(np.arange(len(x)))
            x_df.columns = np.arange(len(x_df.columns))
            return x_df
        elif isinstance(x, pd.Series):
            x_df = x.to_frame()
            x_df.columns = np.arange(len(x_df.columns))
            return x_df
        elif isinstance(x, np.ndarray):
            x_df = pd.DataFrame(x)
            x_df.columns = np.arange(len(x_df.columns))
            return x_df
        elif isinstance(x, list):
            x_df = pd.DataFrame(x)
            x_df.columns = np.arange(len(x_df.columns))
            return x_df
        else:
            raise ValueError(f"The given data of type {type(x)} is not supported.")

    @staticmethod
    def is_float(value):
        """
        Checks if the value is a float in python.

        :param value: The value to check for.
        :return: True, if the value is a python float, False if not.
        """
        try:
            float(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def get_non_float_columns(x):
        """
        Returns all columns indices that are not of float type.
        :param x: The data to check for.
        :return: A python lost of the indices of the columns that are not of float ype.
        """
        x = DataUtils.ensure_pandas_dataframe(x)
        cols = []
        for col in x.columns:
            if not DataUtils.is_float(x[col][0]):
                cols.append(col)

        return cols
