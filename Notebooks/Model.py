import numpy as np
import joblib
import sklearn


class Model:
    """
    Class to tune, save and evaluate machine learning model.
    """

    def __init__(self, model_object, data, target):
        """
        Construct a new Model class for given data and model object

        Parameters:
          model_object (scikit-learn estimator object) : Sci-kit learn model object
          data (pandas.DataFrame) : The dataset containing input and labels
          target (str) : Name of the label column
        """
        self._model = model_object
        self._data = data
        self._target = target
        self._X = self._data.drop(target, axis=1)
        self._y = self._data[[target]]

    def tuned_model(self, k_fold, param_grid, scoring_metrics, primary_scoring_metric):
        """
        Identifies the best model using grid-search cross validation.

        Parameters:
          k_fold (cross-validation generator) : Sci-kit learn cross validation generator object
          param_grid (dict) : Dictionary with parameters names (str) as keys
                              and lists of parameter settings to try as values
          scoring_metrics (str or list) : List of scoring metrics to evaluate performance of cross-validated model
          primary_scoring_metric (str) : If multiple metrics in scoring_metrics, this one will be used as a main to rank models
        """
        self._kfold = k_fold
        self._param_grid = param_grid

        self._gs_cv = sklearn.model_selection.GridSearchCV(self._model, self._param_grid, cv=self._kfold,
                                                   scoring=scoring_metrics, refit=primary_scoring_metric,
                                                   verbose=1, return_train_score=True, n_jobs=-1)

        return self._gs_cv.fit(self._X, np.ravel(self._y)), self._gs_cv.best_estimator_

    def save_model(self, model_name):
        """
        Export the best performing model as joblib object

        Parameters:
          model_name (str) : The object will be exported as this name
        """
        print("\nSaving model " + str(model_name))
        print(f"\nModel {model_name} has been saved successfully!")
        joblib.dump(self._gs_cv.best_estimator_, model_name)

    def model_evaluation(self, test_data):
        """
        Evalute model on test_data using MAE, MSE and R2_score.

        Parameters:
          test_data (pandas.DataFrame) : Dataset containg test input and labels
        """
        test_data_x = test_data.drop(self._target, axis=1)
        test_data_y = test_data[[self._target]]
        y_predictions = self._gs_cv.best_estimator_.predict(test_data_x)

        mse = sklearn.metrics.mean_squared_error(test_data_y, y_predictions)
        mae = sklearn.metrics.mean_absolute_error(test_data_y, y_predictions)
        r2_score = sklearn.metrics.r2_score(test_data_y, y_predictions)

        return y_predictions, {"MAE": mae, "MSE": mse, "R2_score": r2_score}
