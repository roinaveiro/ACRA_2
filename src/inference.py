
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
    probs = sample_label(x_mod, clf, mode='evaluate', n_samples=0)
    exp_utility = np.dot( ut, probs.transpose() )
    return np.argmax(exp_utility, axis=0)


def predict_aware(x_mod, sampler, params):
    '''
    Prediction using adversary aware classifier.

    * x_mod -- instance to predict
    * ut -- utility matrix
    * clf -- underlying classifier
    * sampler -- a function to sample from p(x|x')
    * n_samples -- number of MC samples
    '''
    # Get sample from p(x|x')
    original_sample = sampler(x_mod)
    # Compute p(y|x) for all x in original_sample
    original_probabilities = sample_label(original_sample, params["clf"], mode='evaluate')
    # Compute
    exp_utility = np.dot( params["ut"], np.mean(original_probabilities, axis = 0).transpose() )
    # Return index with maximum utility
    return np.argmax(exp_utility, axis=0)



if __name__ == '__main__':

    X, y = get_spam_data("data/uciData.csv")
    X_train, X_test, y_train, y_test = generate_train_test(X, y, q=0.3)
    clf = LogisticRegression()
    clf.fit(X,y)

    params = {
                "l"      : 1,    # Good instances are y=0,1,...,l-1. Rest are bad
                "k"      : 2,     # Number of classes
                "var"    : 0.1,   # Proportion of max variance of betas
                "ut"     :  np.array([[1.0, 0.0],[0.0, 1.0]]), # Ut matrix for defender
                "ut_mat" :  np.array([[0.0, 0.7],[0.0, 0.0]]), # Ut matrix for attacker rows is
                                            # what the classifier says, columns
                                            # real label!!!
                "sampler_star" : lambda x: sample_original_instance_star(x,
                 n_samples=15, rho=1, x=None, mode='sample', heuristic='uniform'),
                 ##
                 "clf" : clf,
                 "tolerance" : 1, # For ABC
                 "classes" : np.array([0,1]),
                 "S"       : np.array([1,3]), # Set of index representing covariates with
                                             # "sufficient" information
                 "X_train"   : X_train
            }

    #ut = np.array([[1,0], [0,1]])
    #x = np.expand_dims(X[0], axis=0)
    sampler1 = lambda x: sample_original_instance_star(x, 15,
         rho=2, x=None, mode='sample', heuristic='uniform')

    sampler2 = lambda x: sample_original_instance(x, 3, params)

    print( predict_unaware(X_test[0], params["ut"], clf) )

    # Predict for one instance
    print(predict_aware(X_test[0], sampler2, params))


    # Predict for more instances
    #n_samples=10
    #rho=2
    #rows = 30 # predict for 10 samples
    # rows = X.shape[0] # predict for all samples
    # num_cores=4 # it depends of the processor
    # vector_labels = Parallel(n_jobs=num_cores)(delayed(predict_heuristic_par)(i,X, ut, clf, n_samples, rho, heuristic='uniform') for i in range(rows))
    # print("Vector with label predictions: ", vector_labels)
    #def predict_heuristic_par(i,X,ut,clf,n_samples,rho,heuristic):
    #    return predict_heuristic(X[i], ut, clf, n_samples, rho, heuristic='uniform')
