# imports
import numpy as np
import pandas as pd

def lognorm_peak(mu, sigma):
    '''Find the peak of a lognormal distribution, given mu and sigma'''
    return(np.exp(mu-sigma))

def estimate_MAP(samples, *args):
    '''Estimate the MAP values
    Input:
    ---
    samples: pandas dataframe
        dataframe of MCMC traces
    *args: strings
        names of parameters that are fit in the model, e.g. width, center, power_index

    Return:
    ---
    MAP_df: pandas dataframe
        dataframe with columns: parameter (from the input *args), lower_unc, value, upper_unc. The MAP of each parameter is value - lower unc + upper_unc'''

    q = samples.quantile([0.16,0.50,0.84], axis=0)
    MAP_df = pd.DataFrame({'parameter':[],'lower_unc':[],'MAP':[],'upper_unc':[]})
    for arg in args:
        lower, value, upper = q[arg]
        dict_it = {'parameter':[arg],'lower_unc':[value-lower],'MAP':[value],'upper_unc':[value+upper]}
        MAP_df = MAP_df.append(pd.DataFrame(dict_it))
    return(MAP_df.reset_index(drop=True))
