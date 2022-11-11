import numpy as np
from sklearn.metrics import f1_score


def permutation_importance(X, y, model):
    perm = {}
    y_true = model.predict_proba(X)[:,1]
    baseline= f1_score(y, y_true)
    for cols in X.columns:
        value = X[cols]
        X[cols] = np.random.permutation(X[cols].values)
        y_true = model.predict_proba(X)[:,1]
        perm[cols] = f1_score(y, y_true) - baseline
        X[cols] = value
    return perm
