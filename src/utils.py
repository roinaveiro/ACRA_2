import numpy as np
import scipy.special

def get_bin_probs(N, rho):
    binoms = np.array( [scipy.special.binom(N, k) for k in range(rho + 1)] )
    return  binoms / np.sum(binoms), np.sum(binoms)

def change_instance(x, set):
    '''
    Change instance flipping the values indicated by the set of indices
    '''
    x_mod = np.copy(x)
    x_mod[set] = np.abs( x_mod[set] - 1 )
    return x_mod

def distance(x, x_mod):
    return np.sum( np.aps(x - x_mod) )
