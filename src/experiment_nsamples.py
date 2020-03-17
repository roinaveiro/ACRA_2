import numpy as np
import pandas as pd
from data import *
from models import *
from samplers import *
from attacks import *
from inference import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt


if __name__ == '__main__':

    n_exp = 10
    n_samples_grid = [1, 5, 10, 20, 50, 100, 150]
    tolerance = 2
    n_cov = 3
    lr = True

    X, y = get_malware_data("data/malware.csv")

    for i in range(n_exp):

        print('Experiment: ', i)

        acc_raw_clean = np.zeros(len(n_samples_grid))
        acc_raw_att = np.zeros(len(n_samples_grid))
        acc_acra_att = np.zeros(len(n_samples_grid))

        X_train, X_test, y_train, y_test = generate_train_test(X, y, q=0.1)

        if lr:
            clf = LogisticRegression(penalty='l1', C=0.1, solver='saga')
            clf.fit(X_train, y_train)
            weights = np.abs(clf.coef_)
            S = []
            for w in weights:
                S.append( (-w).argsort()[:n_cov] )
            S = np.concatenate( S, axis=0 )
            S = np.unique( S )
            print(S)
            #S = (-weights).argsort()[0,:n_cov]
        else:
            clf = RandomForestClassifier()
            clf.fit(X_train,y_train)
            result = permutation_importance(clf, X_test, y_test, n_repeats=10,
            njobs = -1)
            sorted_idx = -(result.importances_mean).argsort()
            S = sorted_idx[:n_cov]

        for j, n_samples in enumerate(n_samples_grid):

            acc_raw_clean[j] = accuracy_score(y_test, clf.predict(X_test))
        
            params = {
                        "l"      : 1,    # Good instances are y=0,1,...,l-1. Rest are bad
                        "k"      : 4,     # Number of classes
                        "var"    : 0.1,   # Proportion of max variance of betas
                        "ut"     :  np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]), # Ut matrix for defender
                        "ut_mat" :  np.array([[0.0,0.7,0.7,0.7],[0.0,0.0,0.0,0.0],
                            [0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]), # Ut matrix for attacker rows is
                                                    # what the classifier says, columns
                                                    # real label!!!
                        "sampler_star" : lambda x: sample_original_instance_star(x,
                         n_samples=40, rho=1, x=None, mode='sample', heuristic='uniform'),
                         ##
                         "clf" : clf,
                         "tolerance" : tolerance, # For ABC
                         "classes" : np.array([0,1,2,3]),
                         "S"       : S, # Set of index representing covariates with
                                                     # "sufficient" information
                         "X_train"   : X_train,
                         "distance_to_original" : 1 # Numbers of changes allowed to adversary
                    }

            X_att = attack_set(X_test, y_test, params)
            acc_raw_att[j] = accuracy_score(y_test, clf.predict(X_att))

            sampler = lambda x: sample_original_instance(x, n_samples= n_samples, params = params)
            pr = parallel_predict_aware(X_att, sampler, params)

            acc_acra_att[j] =  accuracy_score(y_test, pr)

        df = pd.DataFrame({"n_samples":n_samples_grid, "acc_raw_clean":acc_raw_clean,
         "acc_raw_att":acc_raw_att, "acc_acra_att":acc_acra_att})
        print('Writing Experiment ', i)
        name = "results/exp_samples_1/nsamples_exp" + str(i) + ".csv"
        df.to_csv(name, index=False)
