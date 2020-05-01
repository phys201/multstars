# imports
import numpy as np
import pandas as pd
import seaborn as sns
import pymc3 as pm

def log_likelihood(data, mean, sigma):
    '''
    INPUT
    ------
    data: pandas data frame, containing physical separations and errors [AU] for binary systems
    parameters: tuple containing the center and width of a gaussian to fit the distribution of physical separations

    OUTPUT
    ------
    likelihood: value of the likelihood function
    '''
    p_sep = data['SEP_PHYSICAL'].values
    p_sep_e = data['E_SEP_PHYSICAL'].values

    # gaussian
    G1 = 1 / np.sqrt(2*np.pi * (sigma**2 + p_sep_e**2))             # non-exponential component of Gaussian
    G2 = (-(p_sep-mean)**2 / (2*(sigma**2 + p_sep_e**2))).sum()     # exponential component
    return np.log10(G1.prod()) + G2



def pymc3_fit(data, nsteps=10000, center_max=1000):
    p_sep = data['SEP_PHYSICAL'].values
    p_sep_e = data['E_SEP_PHYSICAL'].values
    
    with pm.Model() as linear_model:
        
        # normal distributions for both priors, centered around expected values
        # this works better with pymc3's sampling than flat priors 
        center = pm.Normal('center', mu=10, sigma=100)   
        width = pm.Normal('width', mu=100, sigma=1000)                           

        # defining the gaussian model for physical separations (in AU)
        sep_physical = pm.Normal('sep_physical', center, width)
        
        sep_catalog = pm.Normal('sep_observed', mu=sep_physical, sigma=p_sep_e, observed=p_sep)
        
        traces = pm.sample(tune=nsteps, draws=nsteps, step=None, chains=1)

    df = pm.trace_to_dataframe(traces)
    return df


