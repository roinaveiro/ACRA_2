import numpy as np
import pandas as pd

def get_spam_data(path):
    data = pd.read_csv(path)
    X = data.drop("spam", axis=1).values
    y = data.spam.values
    return X,y

"""
Generate train and test sets with test set size of q*data_size
"""
def generate_train_test(X, y, q):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=q)
    return X_train, X_test, y_train, y_test
