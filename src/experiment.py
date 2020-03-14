import numpy as np
import pandas as pd
from data import *
from models import *
from samplers import *
from attacks import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt


if __name__ == '__main__':

    lr = True
    X, y = get_malware_data("data/malware.csv")
    X_train, X_test, y_train, y_test = generate_train_test(X, y, q=0.1)
    n=6
    print(X_train.shape)

    if lr:
        clf = LogisticRegression(penalty='l1', C=0.1, solver='saga')
        clf.fit(X_train,y_train)
        weights = np.abs(clf.coef_)
        print(weights.shape)
        S = (-weights).argsort()[0,:n]
        print(S)
    else:

        clf = RandomForestClassifier()
        clf.fit(X_train,y_train)
        result = permutation_importance(clf, X_test, y_test, n_repeats=10,
                                random_state=42, n_jobs=4)
        sorted_idx = result.importances_mean.argsort()


    print('Adversary Unaware Accuracy Clean Data', accuracy_score(y_test, clf.predict(X_test)))

    params = {
                "l"      : 1,    # Good instances are y=0,1,...,l-1. Rest are bad
                "k"      : 4,     # Number of classes
                "var"    : 0.1,   # Proportion of max variance of betas
                "ut"     :  np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]), # Ut matrix for defender
                "ut_mat" :  np.array([[0.0,0.2,0.4,0.6],[0.0,0.0,0.0,0.0],
                    [0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]), # Ut matrix for attacker rows is
                                            # what the classifier says, columns
                                            # real label!!!
                "sampler_star" : lambda x: sample_original_instance_star(x,
                 n_samples=40, rho=2, x=None, mode='sample', heuristic='uniform'),
                 ##
                 "clf" : clf,
                 "tolerance" : 2, # For ABC
                 "classes" : np.array([0.0,1.0,2.0,3.0]),
                 "S"       : S, # Set of index representing covariates with
                                             # "sufficient" information
                 "X_train"   : X_train,
                 "distance_to_original" : 2 # Numbers of changes allowed to adversary
            }

    X_att = attack_set(X_test, y_test, params)
    print('Adversary Unaware Accuracy Attacked Data', accuracy_score(y_test, clf.predict(X_att)))

    #sampler2 = lambda x: sample_original_instance(x, n_samples= 10, params = params)
    #pr = parallel_predict_aware(X_att, sampler2, params)

    #print('Adversary Aware Accuracy Attacked Data', accuracy_score(y_test, pr) )



    '''
    # Read wine dataset
    data = pd.read_csv("data/winequality-white.csv", sep = ";")
    X = data.loc[:, data.columns != "quality"]
    y = data.quality
    ##
    pca = PCA(n_components=X.shape[1], svd_solver='full')
    pca.fit(X)
    X = pca.fit_transform(X)
    ##
    start = 0.01
    stop = 1.0
    grid_size = 10
    #MEAN_GRID = np.logspace(np.log10(start), np.log10(stop), num=grid_size)
    MEAN_GRID = [0.8, 0.9, 1.0]
    #MEAN = 0.5
    VAR = 0.5
    #VAR_GRID = np.logspace(np.log10(start), np.log10(stop), num=grid_size)
    #VAR_GRID = [0.7, 0.8, 0.9, 1.0]
    ##
    N_EXP = 10 # For hold-out validation
    ##
    rmse_raw_clean = np.zeros(N_EXP)
    rmse_nash_clean = np.zeros(N_EXP)
    rmse_bayes_clean = np.zeros(N_EXP)
    rmse_raw_at = np.zeros(N_EXP)
    rmse_nash_at = np.zeros(N_EXP)
    rmse_bayes_at = np.zeros(N_EXP)
    ##
    for MEAN in MEAN_GRID:
    #for VAR in VAR_GRID:
        for i in range(N_EXP):
            status = "MEAN: " + str(MEAN) + " VAR: " + str(VAR) + " EXP: " + str(i)
            print(status)
            X_train, y_train, X_test, y_test = create_train_test(X,y)
            m = torch.distributions.Gamma(torch.tensor([MEAN**2/VAR]), torch.tensor([MEAN/VAR])) ## shape, rate
            ## Parameters
            params = {
                "epochs_rr"    : 350,
                "lr_rr"        : 0.01,
                "lmb"          : 0.0,
                "c_d_train"    : torch.ones([len(y_train), 1]) * MEAN,
                "z_train"      : torch.zeros([len(y_train),1]),
                "c_d_test"     : torch.ones([len(y_test), 1]) * MEAN,
                "z_test"       : torch.zeros([len(y_test),1]),
                "outer_lr"     : 10e-6,
                "inner_lr"     : 10e-4,
                "outer_epochs" : 350,
                "inner_epochs" : 200,
                "n_samples"    : 20,
                "prior"        : m
            }
            ##
            with timer(tag='raw'):
                w_rr = train_rr(X_train, y_train, params)
            ##
            with timer(tag='nash'):
                w_nash = train_nash_rr(X_train, y_train, params)
            ##
            c_d_train_bayes = params["prior"].sample(torch.Size([params["n_samples"], len(y_train)]))#.to("cuda")
            with timer(tag='bayes'):
                w_bayes = train_bayes_rr_test(X_train, y_train, c_d_train_bayes, params, verbose = False)

            ############################################################################################
            ###################### RMSE CALCULATION
            ############################################################################################
            c_d_test = params["prior"].sample(torch.Size([1, len(y_test)]))[0]

            X_test_attacked = attack(X_test, w_rr, c_d_test, params["z_test"])
            pred_attacked =  predict(X_test_attacked, w_rr)
            pred_clean    =  predict(X_test, w_rr)
            #
            rmse_raw_clean[i] = rmse( y_test, pred_clean )
            rmse_raw_at[i]    = rmse( y_test, pred_attacked )
            #
            ##
            X_test_attacked = attack(X_test, w_nash, c_d_test, params["z_test"])
            pred_attacked =  predict(X_test_attacked, w_nash)
            pred_clean    =  predict(X_test, w_nash)
            #
            rmse_nash_clean[i] = rmse( y_test, pred_clean )
            rmse_nash_at[i]    = rmse( y_test, pred_attacked )
            ##
            X_test_attacked = attack(X_test, w_bayes, c_d_test, params["z_test"])
            pred_attacked =  predict(X_test_attacked, w_bayes)
            pred_clean    =  predict(X_test, w_bayes)
            #
            rmse_bayes_clean[i] = rmse( y_test, pred_clean )
            rmse_bayes_at[i]    = rmse( y_test, pred_attacked )
            #####


            df = pd.DataFrame({"EXP":range(N_EXP), "raw_cleandata":rmse_raw_clean, "raw_atdata":rmse_raw_at,
                               "nash_rawdata":rmse_nash_clean, "nash_atdata":rmse_nash_at, "bayes_rawdata":rmse_bayes_clean,
                               "bayes_atdata":rmse_bayes_at})

            name = "results/exp4/"+"mean"+str(MEAN)+"var"+str(VAR)+".csv"
            df.to_csv(name, index=False)
        '''
