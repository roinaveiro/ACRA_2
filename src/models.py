import numpy as np
from sklearn.linear_model import LogisticRegression

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


    
