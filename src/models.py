import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

'''
Discriminative models for the ARA approach to Adversarial Classification
'''

def create_logistic_regression_model(
        penalty,
        dual,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        random_state=None
):
    '''create a logistic regression model 

    penalty -- norm used in the penalization (string)
    dual -- dual or primal formulation (boolean)
    C -- inverse of regulation strength (float)
    fit_intercept -- if a constant should be added to decion function (boolean)
    intercept_scaliing -- if True, instance vector x becomes [x, self.intercept_scaling] (float)
    class_weight -- Over/undersamples the samples of each class given weights (dict, optional)
    random_state -- Seed (int)
    tol -- tolerance (float)

    return the logistic regression model
    '''

    clf_LR = LogisticRegression(
        penalty=penalty,
        dual=dual,
        C=C,
        fit_intercept=fit_intercept,
        intercept_scaling=intercept_scaling,
        random_state=random_state)
    return clf_LR

def logistic_regression_predict(X,clf):
    '''Obtain probability estimates of logistic regression model
    X -- Samples (ndarray)
    clf -- logistic regression model
    return a ndarray with the probability estimates of the samples given
    '''
    return clf.predict_proba(X)

def create_random_forest_model(
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.,
        max_features="auto",
        max_leaf_nodes=None,
        min_impurity_decrease=0.,
        min_impurity_split=1e-7,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None
        ):
    '''create a random forest model
    
    n_estimators -- number of trees (integer),
    criterion -- measures the quality of a split ("gini","entropy"),
    max_depth -- maximum depth of the tree (integer),
    min_samples_split -- minimum samples to split (integer),
    min_samples_leaf -- minimum samples to be a leaf node (integer),
    min_weight_fraction_leaf -- minimum weight fraction of the sum total of weights to be a leaf node (float),
    max_features -- number of features (integer,float,string),
    max_leaf_nodes -- best nodes are defined as relative reduction in impurity (integer),
    min_impurity_decrease -- when a node is split, this is the value of impurity boundary (integer),
    min_impurity_split -- threshold of stopping in tree growth (float),
    bootstrap -- whether bootstrap samples are used to build a tree (boolean),
    oob_score -- whether to use out-of-bag samples to estimate the generalization accuracy (boolean),
    n_jobs -- number of jobs in parallel (integer),
    random_state -- randomness of the bootstrapping of the samples (integer),
    verbose -- controls the verbosity when fitting and predicting (integer),
    warm_start -- with True reuse the solution of the previous call to fit otherwise fit a whole new forest (boolean),
    class_weight -- weights associated with classes (dict, list of dicts)

    '''
    clfRF = RandomForestClassifier(
    n_estimators=n_estimators,
    criterion=criterion,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    min_weight_fraction_leaf=min_weight_fraction_leaf,
    max_features=max_features,
    max_leaf_nodes=max_leaf_nodes,
    min_impurity_decrease=min_impurity_decrease,
    min_impurity_split=min_impurity_split,
    bootstrap=bootstrap,
    oob_score=oob_score,
    n_jobs=n_jobs,
    random_state=random_state,
    verbose=verbose,
    warm_start=warm_start,
    class_weight=class_weight)
    return clfRF

def random_forest_predict(X,clf):
    '''Obtain probability estimates of random forest model
    X -- Samples (ndarray)
    clf -- random forest model
    return a ndarray with the probability estimates of the samples given
    '''
    return clf.predict_proba(X)

    

    
