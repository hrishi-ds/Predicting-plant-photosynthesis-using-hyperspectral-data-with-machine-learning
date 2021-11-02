import numpy as np


class BootstrapFeatureImportance:
    """
    Class to calculate bootstrapped feature importance.
    """

    def __init__(self, data, target):
        """
        Construct a new BootstrapFeatureImportance class for given data.

        Parameters:
          data (pandas.DataFrame) : The dataset containing input and labels
          target (str) : Name of the label column
        """
        self._data = data
        self._target = target
        self._X = self._data.drop(columns=[self._target])
        self._y = self._data[[self._target]]

    def bootstrapped_feature_importance(self, model, num_bootstrap_samples, num_bootstrap_iterations):
        """
        Calculates bootstrapped feature importance and returns dictionary of the feature importances.

        Parameters:
          model (scikit-learn estimator object) : Sci-kit learn Model object
          num_bootstrap_samples (int) : Sample of data with this size will be used in each bootstrap iteration
          num_bootstrap_iterations (int) : Total number of bootstrap iteration to be performed
        """
        X_columns_list = self._X.columns.tolist()

        feature_importance_dict = {x: [] for x in X_columns_list}

        for iteration in range(num_bootstrap_iterations):
            print("Bootstrapping", iteration + 1, ".....")
            self._model = model

            bootstrapped_sample = self._data.sample(n=num_bootstrap_samples, replace=False,
                                                    random_state=iteration)

            bootstrapped_sample_X = bootstrapped_sample.drop(columns=[self._target])
            bootstrapped_sample_y = bootstrapped_sample[[self._target]].values.ravel()

            self._model.fit(bootstrapped_sample_X, bootstrapped_sample_y)

            feature_importance = self._model.feature_importances_

            t_dict = dict(zip(X_columns_list, feature_importance))

            for x in feature_importance_dict:
                feature_importance_dict[x].append(t_dict[x])

        feature_importance_dict_mean = {feature: np.mean(feature_importance_dict.get(feature)) for feature in
                                        feature_importance_dict}
        return feature_importance_dict_mean
