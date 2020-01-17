import numpy as np

'''
Inference functions for the ARA approach to Adversarial Classification
'''


def predict(..., x_mod):
    '''
    Prediction using adversary unaware classifier
    '''
    return y



def predict(..., x_mod, N):
    '''
    Prediction using adversary aware classifier
    '''
    x = sample_original_instance(..., x_mod, n_samples=N)
    return y
