# imports
import numpy as np

# limit for high contrast ratio and low separations
# this is a rough approximation made by eye - should eventually be updated
def icr_max(separation):
    '''returns the inverse of the maximum contrast ratio which can be detected at a given separations'''
    return 1 / (1.8 * np.log(separation) + 3.8)

def sep_min(icr):
    '''returns the maxmimum separation that can be detected for a given inverse contrast ratio'''
    cr = 1/icr
    return np.exp((cr-3.8)/1.8)
