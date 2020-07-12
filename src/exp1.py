import numpy as np
from data import *
from models import *
# from samplers import *
from attacks import *
from utils import *
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


def exp1():
    # Read dataset
    print("Reading dataset...")
    X, y = get_spam_data("../data/uciData.csv")
    X_train, X_test, y_train, y_test = generate_train_test(X, y, q=0.3)

    # Attack dataset
    print("Attacking dataset...")
    clf = LogisticRegression(penalty='l2', C=0.01)
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


    attack = lambda x, y: attack_ARA(x, y, params)
    
    X_att = attack_set(X_test, y_test, attack)
    print(X_att)
    print(X_att.shape)

    # Obtain labels with ACRA2
    print("Predicting...")
    sampler1 = lambda x: sample_original_instance_star(x, 15, rho=2, x=None, mode='sample', heuristic='uniform')
    sampler2 = lambda x: sample_original_instance(x, n_samples= 3, params = params)
    pr = parallel_predict_aware(X_test[1:5], sampler2, params)
    y_clean = clf.predict(X_test[1:5])
    
    # Check accuracy
    print("ACC LR:", accuracy_score(y_test, y_clean))
    print("ACC ACRA:", accuracy_score(y_test, pr))





if __name__ == '__main__':

    exp1()




