import numpy as np
import pandas as pd
from sklearn.model_selection import *


class DataPreparation:
    """class containing methods for data preparation"""

    def __init__(self, data, target):
        """
        Construct a new data preparation class with given data and target column

        Parameters:
          data (pandas.DataFrame) : The dataset containing input and labels
          target (str) : Name of the label column
        """

        self._data = data
        self._X = self._data.drop(target, axis=1)
        self._y = self._data[[target]]
        self._target = target

    def digitize_column(self, colName):
        """
        Returns a digitised version of the given column from the dataset

        Parameters:
          colName (str): The name of the column to be digitised
        """
        bins = np.percentile(self._data[colName], [0, 25, 50, 75])
        return np.digitize(self._data[colName], bins)

    def train_test_split(self, test_split_percent, seed, stratify=None):
        """
        Generate a train-test split of the dataset.

        Parameters:
          test_split_percent (float) : The proportion of data to be split into test
          seed (int) : Seed for reproducibility of the random splits
          stratify (array-like), default=None : Split will be stratified based on this array.

        """
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._X,
                                                                                    self._y,
                                                                                    random_state=seed,
                                                                                    test_size=test_split_percent,
                                                                                    stratify=stratify)
        self._X_test = pd.DataFrame(self._X_test, columns=self._X.columns)
        self._y_test = pd.DataFrame(self._y_test, columns=[self._target])

        self._test = pd.concat([self._X_test, self._y_test], axis=1)

        self._X_train = pd.DataFrame(self._X_train, columns=self._X.columns)
        self._y_train = pd.DataFrame(self._y_train, columns=[self._target])

        self._train = pd.concat([self._X_train, self._y_train], axis=1)

        return self._train, self._test
