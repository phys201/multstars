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



# function to fit the model (from week 9 seciton notebook)
def pymc3_fit(data, nsteps=10000, center_max=1000, metropolis=False):
    
    with pm.Model() as linear_model:
        center = pm.Uniform('center', lower=0, upper=center_max)  # Continuous uniform log-likelihood
        width = pm.HalfFlat('width')                   # Improper flat prior over the positive reals (Siva & Skilling 2006)

        loglike = log_likelihood(data, center, width)
        pm.Potential('obs', loglike)
        
        step = pm.Metropolis() if metropolis else None
        traces = pm.sample(tune=nsteps, draws=nsteps, step=step, chains=1)

    df = pm.trace_to_dataframe(traces)
    return df


