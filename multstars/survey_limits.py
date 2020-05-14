# imports
import numpy as np

# LIMITS FOR WHAT WILL APPEAR IN THE FINAL DATA
# these can be constants or functions of each other


# limits for separations
def sep_min(contrast):
    '''returns the minimum separation [arcseconds] that can be detected for a given contrast ratio'''
    return np.exp((contrast-3.8)/1.8)

def sep_max(contrast):
    '''returns the maximum separation [arcseconds] that can be detected for a given contrast ratio'''
    return 4


# limits for high contrast ratios
def cr_max(separation):
    '''returns the maximum contrast ratio which can be detected at a given separation [arcseconds]'''
    return (1.8 * np.log(separation) + 3.8)

def cr_min(separation):
    '''returns the minimum contrast ratio which can be detected at a given separation [arcseconds]'''
    return 0.01
