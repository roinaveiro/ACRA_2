import numpy as np

"""
Prediction using adversary unaware classifier
"""
def predict(..., x_mod):
    return y


"""
Prediction using adversary aware classifier
"""
def predict(..., x_mod, N):
    x = sample_original_instance(..., x_mod, n_samples=N)
    return y
