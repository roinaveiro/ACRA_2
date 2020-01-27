import numpy as np
import pandas as pd
from data import *
from utils import *

'''
Relevant samplers for the ARA approach to Adversarial Classification
'''


def sample_instance(n_samples):
    '''
    Get n_samples from p(x)
    '''
    return


def sample_transformed_instance(x, y, n_samples, mode):
    '''
    Sample or evaluate p(x'|x,y)
    * If mode is "sample", a sample is obtained
    * If mode is "evaluate", probability is computed and returned
    '''
    return


def sample_label(X,clf,ut, mode):
    '''
    Sample or evaluate p(y|x)
    * If mode is "sample", a sample is obtained (mode==1)
    * If mode is "evaluate", probability is computed and returned (mode==2)
    X -- dataset (ndarray)
    clf -- your favority classifier (obj)
    ut -- utility matrix
    '''
    if mode==1:
        aux_matrix = np.dot(ut,clf.predict_proba(X,clf).transpose())
        return (np.argmax(aux_matrix,axis=0))
    else:
        return np.dot(ut,clf.predict_proba(X,clf).transpose())


def sample_original_instance(x_mod, n_samples):
    '''
    Sample or evaluate p(x|x')
    '''

    return


def sample_utility(alpha, beta, n_samples):
    '''
    Sample a utility for ARA
    '''
    ut = np.random.beta(alpha, beta, n_samples)
    return ut

def sample_probability(n_samples):
    '''
    Sample a probability for ARA
    '''
    pass
    #prob = np.random.beta(..., ..., n_samples)
    #return prob

def sample_original_instance_star(x_mod, n_samples, rho, x=None, mode='sample', heuristic='uniform'):
    '''
    Sample or evaluate p(x|x') using a metric based approachself.

    * x_mod -- original instance
    * rho -- maximum allowed distance

    '''
    N = x_mod.shape[0]
    if heuristic == 'uniform':
        ##
        pr, tot = get_bin_probs(N, rho)
        if mode == 'sample':
            lengths = np.random.choice( range(rho+1), p=pr, size=n_samples )
            indices = [np.random.choice( range(N), replace = False, size = rho) for rho in lengths]
            samples = np.stack( [change_instance(x_mod, set) for set in indices], axis=0 )
            return samples

        if mode == 'evaluate':
            if distance(x, x_mod) <= rho:
                return 1/tot
            else:
                return 0

        if heuristic == 'penalized_distance':
            pass


        #return

# Some tests

if __name__ == '__main__':

    X, y = get_spam_data("data/uciData.csv")

    x = X[0]
    samples = sample_original_instance_star(x, n_samples=10, rho=10)
    print(samples)

    print( get_bin_probs(3, 2) )
