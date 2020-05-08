# imports
import numpy as np
import pandas as pd
import seaborn as sns
import pymc3 as pm
from pymc3.distributions.dist_math import normal_lcdf, normal_lccdf
import theano.tensor as tt


# useful functions integrate censored data out through the likelihood
# from https://docs.pymc.io/notebooks/censored_data.html

def left_censored_likelihood(mu, sigma, n_left_censored, lower_bound):
    ''' Likelihood of left-censored data. '''
    return n_left_censored * normal_lcdf(mu, sigma, lower_bound)


def right_censored_likelihood(mu, sigma, n_right_censored, upper_bound):
    ''' Likelihood of right-censored data. '''
    return n_right_censored * normal_lccdf(mu, sigma, upper_bound)



# LIMITS FOR WHAT WILL APPEAR IN THE FINAL DATA

# limits for separations [arcsec]
sep_ang_max = 4.0


# limit for contrast ratio [delta mag]
# this is a rough approximation made by eye - should eventually be updated
def cr_max(separation):
    return 1.8 * np.log(separation) + 3.8

# from inverse mass ratios to inverse contrast ratios
# this is an empirical relationship based on the results from (Lamman et al.) Section 4.4
def imr_to_icr(x):
    g = np.poly1d([ 20.34258759, -63.07636188,  72.52402942, -40.42916102, 10.64895881])
    return 1 / g(1/x)


def pymc3_hrchl_fit(data, nsteps=1000):
    
    
    asep = data['asep'].values
    asep_err = data['asep_err'].values
    cr = data['cr'].values
    cr_err = data['cr_err'].values
    parallax = data['parallax'].values


    with pm.Model() as hierarchical_model:

        # PRIORS
        # -------------
        
        # Separations
        center = pm.Gamma('center', mu=20, sigma=10)
        # the Gamma distribution for separations should have 0 < width < center, 
        # because we're assuming that the peak is above 0 for separations
        width_diff = pm.Beta('width_difference', alpha=2, beta=2)
        width = pm.Deterministic('width', center-(center*width_diff))
        
        # power_index must be bound to work with the Kumaraswamy model
        p = pm.Gamma('p', mu=0.2, sigma=.5)
        power_index = pm.Deterministic('power_index', 1+p)
        
        
        # MODELS OF POPULATIONS PHYSICAL PROPERTIES
        # --------------
        
        # Gaussian model for separations (in AU)
        sep_physical = pm.Gamma('sep_physical', mu=center, sigma=width)
        
        # Mass Ratios - inverted
        mass_ratios_inverted = pm.Pareto('mass_ratios_inverted', alpha=power_index, m=1)
        
        
        # MAPPING FROM PHYSICAL TO OBSERVED PROPERTIES
        # ---------------

        #  physical separations to angular separations
        sep_angular = pm.Deterministic('sep_angular', sep_physical * parallax)
        
        # inverted mass ratios to inverted contrast ratios
        contrast_ratios_inverted = pm.Deterministic('contrast_ratios_inverted', imr_to_icr(mass_ratios_inverted))
        
        # LIKELIHOODS, WITH MEASUREMENT ERROR
        # -----------------
        
        # separations
        seps_truncated = pm.Bound(pm.Normal, upper=4)('seps_truncated', mu=sep_angular, sigma=asep_err)
        sep_observed = pm.Normal('sep_observed', mu=seps_truncated, sigma=asep_err, observed=asep)
        
        # contrast ratios
        cr_observed_inverse = pm.Normal('cr_observed', mu=contrast_ratios_inverted, sigma=cr_err/cr, observed=1/cr)
        
        
        # ACCOUNTING FOR OBSERVATION LIMITS
        # ------------------
        
        ##truncated_crs = pm.TruncatedNormal('crs_trunc', mu=cr_observed_inverse, sigma=cr_err*cr_observed_inverse, lower=1/7)

        
        
        # RUNNING THE FIT
        # -------------------
        traces = pm.sample(tune=nsteps, draws=nsteps, step=None, chains=1)
        
        # output as dataframe
        df = pm.trace_to_dataframe(traces)
    return df