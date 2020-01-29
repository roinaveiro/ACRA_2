
import numpy as np
import pandas as pd
from data import *
from models import *
from samplers import *
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed

'''
Inference functions for the ARA approach to Adversarial Classification
'''


def predict_unaware(x_mod, ut, clf):
    '''
    Prediction using adversary unaware classifier
    '''
    exp_utility = np.dot( ut, clf.predict_proba(x_mod).transpose() )
    return np.argmax(exp_utility, axis=0)



def predict_aware(x_mod, ut, clf, sampler, n_samples):
    '''
    Prediction using adversary aware classifier.

    * x_mod -- instance to predict
    * ut -- utility matrix
    * clf -- underlying classifier
    * sampler -- a function to sample from p(x|x')
    * n_samples -- number of MC samples
    '''
    
    original_sample = sampler(x_mod, n_samples)

    original_probabilities = sample_label(original_sample, clf,
        n_samples=0, mode='evaluate')

    exp_utility = np.dot( ut, np.mean(original_probabilities, axis = 0).transpose() )

    return np.argmax(exp_utility, axis=0)



if __name__ == '__main__':

    X, y = get_spam_data("data/uciData.csv")
    clf = LogisticRegression()
    clf.fit(X,y)
    ut = np.array([[1,0], [0,1]])
    x = np.expand_dims(X[0], axis=0)

    print( predict_unaware(x, ut, clf) )

    sampler = lambda x, n: sample_original_instance_star(x, n,
        rho=2, x=None, mode='sample', heuristic='uniform')

    # Predict for one instance
    print(predict_aware(X[0], ut, clf, sampler, n_samples=10))
    # Predict for more instances
    #n_samples=10
    #rho=2
    #rows = 30 # predict for 10 samples
    # rows = X.shape[0] # predict for all samples
    # num_cores=4 # it depends of the processor
    # vector_labels = Parallel(n_jobs=num_cores)(delayed(predict_heuristic_par)(i,X, ut, clf, n_samples, rho, heuristic='uniform') for i in range(rows))
    # print("Vector with label predictions: ", vector_labels)
