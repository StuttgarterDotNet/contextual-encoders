import pandas as pd
import numpy as np


class DataUtils:
    @staticmethod
    def ensure_pandas_dataframe(x):
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
        try:
            float(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def get_non_float_columns(x):
        x = DataUtils.ensure_pandas_dataframe(x)
        cols = []
        for col in x.columns:
            if not DataUtils.is_float(x[col][0]):
                cols.append(col)

        return cols
