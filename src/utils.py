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

def get_beta_parameters(mu, k):
    '''
    * mu -- mean of the beta distribution
    * k -- proportion of maximum variance to have convex beta
    '''
    var = k * mu * min(mu * (1.0 - mu) / (1.0 + mu) , (1.0 - mu)**2 / (2.0 - mu))
    alpha = ( (1-mu) / var - 1 / mu ) * mu**2
    beta = alpha * ( 1/mu - 1 )

    return alpha, beta
'''
def deltas(ra, var):

    deltas = np.zeros((len(ra),2))

    for i in range(len(ra)):
        s2 = var *ra[i]* min(ra[i] * (1.0 - ra[i]) / (1.0 + ra[i]) , \
                             (1.0 - ra[i])**2 / (2.0 - ra[i]))     ## proportion of maximum
                                                                #variance of convex beta
        deltas[i][0] = ( ( 1.0 - ra[i] ) / s2 - 1.0 / ra[i]) * ra[i]**2
        deltas[i][1] = deltas[i][0] * ( 1.0/ra[i] - 1.0 )

    return(deltas)
'''
