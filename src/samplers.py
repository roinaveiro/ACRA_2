import numpy as np
import pandas as pd
from data import *
from utils import *
from models import *
from sklearn.linear_model import LogisticRegression

'''
Relevant samplers for the ARA approach to Adversarial Classification
'''

def sample_original_instance(x_mod, n_samples):
    '''
    ABC function to sample from p(x|x')
    '''
    tolerance = params["tolerance"]
    samples = np.array([n_samples, x_mod.shape[0]])
    for i in range(n_samples):
        x = sample_instance()
        probs = sample_label(x, clf)[0]
        dist = params["tolerance"] + 1
        while dist > tolerance:
            y = np.random.choice(params["classes"], p=probs)
            x_tilde = sample_transformed_instance(x, y, params)
            dist = distance(x_tilde, x_mod)
        samples[i] = x_tilde
    return samples

def sample_instance(n_samples=1):
    '''
    Get n_samples from p(x)
    '''
    return



def sample_transformed_instance(x, y, params):
    '''
    For ABC, sample just one instance of p(x'|x,y)
    * Good labels are indexed as l, l+1, ..., k
    * If mode is "sample", a sample is obtained
    * If mode is "evaluate", probability is computed and returned
    '''
    l = params["l"]
    if y < l:
        return x
    else:
        uts = sample_utility(y, params)
        perturbations = original_instances_given_dist(x, n=2)
        prob_matrix = np.zeros([perturbations.shape[0], l])
        ##
        for i in range(perturbations.shape[0]): ## ESTO ES UN CHOCHO
            prob_matrix[i] = sample_probability(x, params)
        ##
        expected_ut = np.dot(prob_matrix, uts)
        idx = np.argmax(expected_ut)
        return perturbations[idx]


def sample_label(X, clf, mode='evaluate', n_samples=0):
    '''
    Sample or evaluate p(y|x)
    * If mode is 'sample', a sample is obtained
    * If mode is 'evaluate', probability is computed and returned
    X -- dataset (ndarray)
    clf -- classifier (obj)
    n_samples -- number of samples to get
    '''
    if X.ndim == 1:
        X = np.expand_dims(X, 0)
    if mode == 'evaluate':
        return clf.predict_proba(X)

    if mode == 'sample':
        pass


def sample_utility(i, params):
    '''
    Sample a utility for ARA

    * c -- label predicted by classifier
    * i -- real label of instance
    * attacker_ut -- utility matrix with samplers
    * n_samples -- number of samples to get

    '''
    l = params["l"]
    assert i >= l,  "Class is good"
    ut_mat = params["ut_mat"]
    ut_samples = np.zeros([params["l"],1])
    var = params["var"]
    for j in range(l):
        alpha, beta = get_beta_parameters(ut_mat[j, i], var)
        ut_samples[j] = np.random.beta(alpha, beta)

    return ut_samples

def sample_probability(x, params):
    '''
    Sample a probability for ARA. DOUBLE-CHECK
    '''
    l = params["l"]
    var = params["var"]
    prob_samples = np.zeros(params["l"])
    for c in range(l):
        # Sample from p^*(x|x')
        sample = params["sampler_star"](x)
        # Compute p(y|x) for each x in sample
        probs = sample_label(sample, params["clf"],
            mode='evaluate', n_samples=0)[:,c]
        # Approximate with MC the value of the mean
        mean = np.mean(probs, axis = 0)
        # Get parameters of beta distribution
        alpha, beta = get_beta_parameters(mean, var)
        prob_samples[c] = np.random.beta(alpha, beta)

    return prob_samples

def sample_original_instance_star(x_mod, n_samples, rho, x=None, mode='sample', heuristic='uniform'):
    '''
    Sample or evaluate p(x|x') using a metric based approach.

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
    clf = LogisticRegression()
    clf.fit(X,y)

    params = {
                "l"      : 1,    # Good instances are y=0,1,...,l-1. Rest are bad
                "k"      : 2,     # Number of classes
                "var"    : 0.1,   # Proportion of max variance of betas
                "ut_mat" :  np.array([[0.0, 0.7],[0.0, 0.0]]), # Ut matrix for attacker rows is
                                            # what the classifier says, columns
                                            # real label!!!
                "sampler_star" : lambda x: sample_original_instance_star(x,
                 n_samples=15, rho=1, x=None, mode='sample', heuristic='uniform'),
                 "clf" : clf,
                 "tolerance" : 1 # For ABC
                 "classes" : np.array([0,1])
            }

    x = X[0]
    y = y[0]
    probs = sample_label(x, clf)
    ajaja = np.random.choice(np.array([0,1]), p=probs[0])
    print(ajaja)
    #print sample_probability(x, params)
    #print(sample_utility(1, params).shape)
    #print(y)
    #print(sample_transformed_instance(x, y, params))




    if False:
        x = X[0]
        samples = sample_original_instance_star(x, n_samples=10, rho=10)
        ss = sample_label(samples, clf)
        print( np.mean(ss, axis=0)  )

        sampler = lambda x, n: sample_original_instance_star(x, n, rho=10)
        print( sample_probability_c(x, 1, sampler, 10, 100, 0.9) )

    ##
