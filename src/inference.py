
import numpy as np
import pandas as pd
from data import *
from models import *
from samplers import *
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)


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

def parallel_predict_aware(X_test, sampler, params):
    def predict_aware_par(i, X_test, sampler, params):
        return predict_aware(X_test[i], sampler, params)
    ##
    num_cores=4 # it depends of the processor
    preds = Parallel(n_jobs=num_cores)(delayed(predict_aware_par)(i, X_test, sampler, params) for i in range(X_test.shape[0]))
    return np.array(preds)



if __name__ == '__main__':

    X, y = get_spam_data("data/uciData.csv")
    X_train, X_test, y_train, y_test = generate_train_test(X, y, q=0.3)
    clf = LogisticRegression(penalty='l1', C=0.01)
    clf.fit(X,y)
    ## Get "n" more important covariates
    n=5
    weights = np.abs(clf.coef_)
    S = (-weights).argsort()[0,:n]

    params = {
                "l"      : 1,    # Good instances are y=0,1,...,l-1. Rest are bad
                "k"      : 2,     # Number of classes
                "var"    : 0.1,   # Proportion of max variance of betas
                "ut"     :  np.array([[1.0, 0.0],[0.0, 1.0]]), # Ut matrix for defender
                "ut_mat" :  np.array([[0.0, 0.7],[0.0, 0.0]]), # Ut matrix for attacker rows is
                                            # what the classifier says, columns
                                            # real label!!!
                "sampler_star" : lambda x: sample_original_instance_star(x,
                 n_samples=15, rho=2, x=None, mode='sample', heuristic='uniform'),
                 ##
                 "clf" : clf,
                 "tolerance" : 3, # For ABC
                 "classes" : np.array([0,1]),
                 "S"       : S, # Set of index representing covariates with
                                             # "sufficient" information
                 "X_train"   : X_train,
                 "distance_to_original" : 2 # Numbers of changes allowed to adversary
            }

    #ut = np.array([[1,0], [0,1]])
    #x = np.expand_dims(X[0], axis=0)
    sampler1 = lambda x: sample_original_instance_star(x, 15,
         rho=2, x=None, mode='sample', heuristic='uniform')

    sampler2 = lambda x: sample_original_instance(x, n_samples= 3, params = params)

    if False:
        #
        print( predict_unaware(X_test[0], params["ut"], clf) )
        # Predict for one instance
        print(predict_aware(X_test[0], sampler2, params))

    if True:
        pr = parallel_predict_aware(X_test, sampler2, params)
        print(pr)
    # Predict for more instances
    #n_samples=10
    #rho=2
    #rows = 30 # predict for 10 samples
    # rows = X.shape[0] # predict for all samples
    # num_cores=4 # it depends of the processor
    # print("Vector with label predictions: ", vector_labels)
    #def predict_heuristic_par(i,X,ut,clf,n_samples,rho,heuristic):
    #    return predict_heuristic(X[i], ut, clf, n_samples, rho, heuristic='uniform')
