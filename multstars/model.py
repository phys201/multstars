# imports
import numpy as np
import pandas as pd
import seaborn as sns
import pymc3 as pm
from pymc3.distributions.dist_math import normal_lcdf, normal_lccdf

# useful functions integrate censored data out through the likelihood
# from https://docs.pymc.io/notebooks/censored_data.html

def left_censored_likelihood(mu, sigma, n_left_censored, lower_bound):
    ''' Likelihood of left-censored data. '''
    return n_left_censored * normal_lcdf(mu, sigma, lower_bound)


def right_censored_likelihood(mu, sigma, n_right_censored, upper_bound):
    ''' Likelihood of right-censored data. '''
    return n_right_censored * normal_lccdf(mu, sigma, upper_bound)




# observation limitations for separations - later this will be an input parameter
sep_ang_max = 4.0
sep_ang_min = 0.1

def pymc3_hrchl_fit(data, nsteps=1000):
    
    
    asep = data['asep'].values
    asep_err = data['asep_err'].values
    cr = data['cr'].values
    cr_err = data['cr_err'].values
    parallax = data['parallax'].values


    with pm.Model() as hierarchical_model:

        # normal distributions for both priors, centered around expected values
        # this works better with pymc3's sampling than flat priors 
        center = pm.Normal('center', mu=30, sigma=100)   
        width = pm.Normal('width', mu=50, sigma=100)     
        
        # defining the gaussian model for separations (in AU)
        # must be bound because no separations less than 0
        sep_physical = pm.Bound(pm.Normal, lower=0)('sep_physical', center, width)

        # transforming from physical separations to angular separations
        sep_angular = pm.Deterministic('sep_angular', sep_physical * parallax)
        
        # likelihood
        sep_observed = pm.Normal('sep_observed', mu=sep_angular, sigma=asep_err, observed=asep)
        
        
        # RVs for the number of data points which fall above and below the observational range
        n_low = pm.HalfFlat('n_low')
        n_high = pm.HalfFlat('n_high')
        
        # integrate out the points which fall outside of the observation limits from the likelihood
        left_censored = pm.Potential('left_censored', left_censored_likelihood(sep_observed, asep_err, n_low, sep_ang_min))
        right_censored = pm.Potential('right_censored', right_censored_likelihood(sep_observed, asep_err, n_high, sep_ang_max))

        
        # running the fit
        traces = pm.sample(tune=nsteps, draws=nsteps, step=None, chains=1)
        
        # output as dataframe
        df = pm.trace_to_dataframe(traces)
    return df