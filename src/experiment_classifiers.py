import numpy as np
import pandas as pd
from data import *
from models import train_clf
from samplers import *
from attacks import *
from inference import *
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


if __name__ == '__main__':

    n_exp = 10
    n_samples = 20
    tolerance = 1
    n_cov = 11
    # flag = 'svm'
    flag_grid = ['nb'] # ['nn', 'svm', 'lr', 'rf']

    X, y = get_spam_data("data/uciData.csv")

    for i in range(n_exp):

        print('Experiment: ', i)

        acc_raw_clean = np.zeros(len(flag_grid))
        acc_raw_att = np.zeros(len(flag_grid))
        acc_acra_att = np.zeros(len(flag_grid))

        X_train, X_test, y_train, y_test = generate_train_test(X, y, q=0.1)

        for j, flag in enumerate(flag_grid):

            clf, S = train_clf(X_train, y_train, n_cov, flag)
            acc_raw_clean[j] = accuracy_score(y_test, clf.predict(X_test))
            print( str(flag) + "accuracy on clean data", acc_raw_clean )

            params = {
                        "l"      : 1,    # Good instances are y=0,1,...,l-1. Rest are bad
                        "k"      : 2,     # Number of classes
                        "var"    : 0.1,   # Proportion of max variance of betas
                        "ut"     :  np.array([[1.0, 0.0],[0.0, 1.0]]), # Ut matrix for defender
                        "ut_mat" :  np.array([[0.0, 0.7],[0.0, 0.0]]), # Ut matrix for attacker rows is
                                                    # what the classifier says, columns
                                                    # real label!!!
                        "sampler_star" : lambda x: sample_original_instance_star(x,
                         n_samples=40, rho=1, x=None, mode='sample', heuristic='uniform'),
                         ##
                         "clf" : clf,
                         "tolerance" : tolerance, # For ABC
                         "classes" : np.array([0,1]),
                         "S"       : S, # Set of index representing covariates with
                                                     # "sufficient" information
                         "X_train"   : X_train,
                         "distance_to_original" : 2, # Numbers of changes allowed to adversary
                         "stop" : True, # Stopping condition for ABC
                         "max_iter": 1 ## max iterations for ABC when stop=True

                    }


            X_att = attack_set(X_test, y_test, params)
            acc_raw_att[j] = accuracy_score(y_test, clf.predict(X_att))
            print( str(flag) + "accuracy on tainted data", acc_raw_att)

            sampler = lambda x: sample_original_instance(x, n_samples= n_samples, params = params)
            pr = parallel_predict_aware(X_att, sampler, params)
            acc_acra_att[j] =  accuracy_score(y_test, pr)
            print( str(flag) + "ACRA accuracy on tainted data", acc_acra_att)

        df = pd.DataFrame({"classifier":flag_grid, "acc_raw_clean":acc_raw_clean,
         "acc_raw_att":acc_raw_att, "acc_acra_att":acc_acra_att})
        print('Writing Experiment ', i)
        name = "results/exp_classifiers/" + "nb_good_exp_classifiers" + str(i) + ".csv"
        df.to_csv(name, index=False)
