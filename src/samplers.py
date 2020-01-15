import numpy as np

"""
Relevant samplers for the ARA approach to Adversarial Classification
"""


"""
Get n_samples from p(x)
"""
def sample_instance(..., n_samples):
    return

"""
Sample or evaluate p(x'|x,y)
* If mode is "sample", a sample is obtained
* If mode is "evaluate", probability is computed and returned
"""
def sample_transformed_instance(..., x, y, n_samples, mode):
    return

"""
Sample or evaluate p(y|x)
* If mode is "sample", a sample is obtained
* If mode is "evaluate", probability is computed and returned
"""
def sample_label(..., x, n_samples, mode):
    return

"""
Sample or evaluate p(x|x')
"""
def sample_original_instance(..., x_mod, n_samples):
    return


"""
Sample a utility for ARA
"""
def sample_utility(..., n_samples):
    return

"""
Sample a probability for ARA
"""
def sample_utility(..., n_samples):
    return
