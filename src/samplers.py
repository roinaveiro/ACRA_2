import numpy as np

'''
Relevant samplers for the ARA approach to Adversarial Classification
'''


def sample_instance(n_samples):
    '''Get n_samples from p(x)
    '''
    return


def sample_transformed_instance(x, y, n_samples, mode):
    '''
    Sample or evaluate p(x'|x,y)
    * If mode is "sample", a sample is obtained
    * If mode is "evaluate", probability is computed and returned
    '''
    return


def sample_label(X,clf,n_samples, mode):
    '''
    Sample or evaluate p(y|x)
    * If mode is "sample", a sample is obtained (mode==1)
    * If mode is "evaluate", probability is computed and returned (mode==2)
    X -- dataset (ndarray)
    clf -- your favority classifier (obj)
    n_samples -- number of samples to get 
    '''
    if mode==1:
        predictProbaValues = clf.predict_proba(X)
        np.repeat(np.atleast_3d(predictProbaValues),n_samples,2)
        return np.repeat(np.atleast_3d(predictProbaValues),n_samples,2)
    else:
        return clf.predict_proba(X)


def sample_original_instance(x_mod, n_samples):
    '''
    Sample or evaluate p(x|x')
    '''

    return


def sample_utility(n_samples):
    '''
    Sample a utility for ARA
    '''
    return


def sample_utility(n_samples):
    '''
    Sample a probability for ARA
    '''
    return
