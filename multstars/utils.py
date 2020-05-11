# imports

import numpy as np

def lognorm_peak(mu, sigma):
    '''Find the peak of a lognormal distribution, given mu and sigma'''
    return(np.exp(mu-sigma))

# limit for high contrast ratio and low separations
# this is a rough approximation made by eye - should eventually be updated
def icr_max(separation):
    '''returns the inverse of the maximum contrast ratio which can be detected at a given separations'''
    return 1 / (1.8 * np.log(separation) + 3.8)

def sep_min(icr):
    '''returns the maxmimum separation that can be detected for a given inverse contrast ratio'''
    cr = 1/icr
    return np.exp((cr-3.8)/1.8)

# from inverse mass ratios to inverse contrast ratios
# this is an empirical relationship based on the results from (Lamman et al.) Section 4.4
def imr_to_icr(x):
    g = np.poly1d([ 20.34258759, -63.07636188,  72.52402942, -40.42916102, 10.64895881])
    return 1 / g(1/x)
