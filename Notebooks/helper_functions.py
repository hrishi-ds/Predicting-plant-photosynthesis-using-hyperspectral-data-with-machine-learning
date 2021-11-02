
def k_folds(splits_n, rnd_state):
    """
    Creates stratified k-fold object with given splits.

    Parameters:
      splits_n (int) : Number of splits to bes used in k-fold
      rnd_state (int) : Seed for reproducibility of k-fold indices
    """
    from sklearn.model_selection import StratifiedKFold
    k_fold_train = StratifiedKFold(n_splits=splits_n, shuffle=True, random_state=rnd_state)
    return k_fold_train


def pred_ints(model, X, percentile=95):
    """
    Calculates prediction interval of random forest model prediction

    Parameters:
      model (scikit-learn estimator object) : Sci-kit learn Model object
      X (pandas DataFrame): Dataframe of test inputs
      percentile (int) : Percentile for error bound
    """
    import numpy as np
    err_down = []
    err_up = []
    for x in X.values:
        preds = []
        for pred in model.estimators_:
            preds.append(pred.predict(np.reshape(x, (1, -1))))
        err_down.append(np.percentile(preds, (100 - percentile) / 2.))
        err_up.append(np.percentile(preds, 100 - (100 - percentile) / 2.))
    return np.array(err_down), np.array(err_up)
