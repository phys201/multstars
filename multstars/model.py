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


# limit for contrast ratio
# this is a rough approximation made by eye - should eventually be updated
def icr_max(separation):
    return 1 / (1.8 * np.log(separation) + 3.8)

# from inverse mass ratios to inverse contrast ratios
# this is an empirical relationship based on the results from (Lamman et al.) Section 4.4
def imr_to_icr(x):
    g = np.poly1d([ 20.34258759, -63.07636188,  72.52402942, -40.42916102, 10.64895881])
    return 1 / g(1/x)


def pymc3_hrchl_fit(data, nsteps=1000):
    
    
    asep = data['asep'].values
    asep_err = data['asep_err'].values
    cr = np.abs(data['cr'].values)
    cr_err = data['cr_err'].values
    parallax = data['parallax'].values
    
    cr_inverse = 1/cr
    cr_err_inverse = cr_err/cr


    with pm.Model() as hierarchical_model:

        # PRIORS
        # -------------
        
        # Separations
        center = pm.Gamma('center', mu=20, sigma=15)
        # the Gamma distribution for separations should have 0 < width < center, 
        # because we're assuming that the peak is above 0 for separations
        width_diff = pm.Beta('width_difference', alpha=2, beta=2)
        width = pm.Deterministic('width', center-(center*width_diff))
        
        # power_index must be bound to work with the Kumaraswamy model
        ##p = pm.Gamma('p', mu=0.3, sigma=.25)
        ##power_index = pm.Deterministic('power_index', 1+p)
        power_index = pm.Normal('power_index', mu=1.2, sigma=.2)
        
        
        # MODELS OF POPULATIONS PHYSICAL PROPERTIES
        # --------------
        
        # Gaussian model for separations (in AU)
        sep_physical = pm.Gamma('sep_physical', mu=center, sigma=width, shape=len(asep))
        
        # Mass Ratios - inverted
        mass_ratios_inverted = pm.Pareto('mass_ratios_inverted', alpha=power_index, m=1)#, shape=len(cr_inverse))
        
        
        # MAPPING FROM PHYSICAL TO OBSERVED PROPERTIES
        # ---------------

        #  physical separations to angular separations
        sep_angular = pm.Deterministic('sep_angular', sep_physical * parallax)
        
        # inverted mass ratios to inverted contrast ratios
        contrast_ratios_inverted = pm.Deterministic('contrast_ratios_inverted', imr_to_icr(mass_ratios_inverted))
        
        
        
        # LIKELIHOODS, WITH MEASUREMENT ERROR
        # -----------------
        
        # separations
        sep_observed = pm.Normal('sep_observed', mu=sep_angular, sigma=asep_err, observed=asep)
        
        # contrast ratios
        cr_observed_inverse = pm.Normal('cr_observed', mu=contrast_ratios_inverted, sigma=cr_err_inverse, observed=cr_inverse)
        
        
        # ACCOUNTING FOR OBSERVATION LIMITS
        # ------------------
        
        # integrate out the points which fall outside of the observation limits from the likelihood
        # pm.Potential accounts for truncation and normalizes likelihood
        
        # separation limits
        truncated_seps_upper = pm.Potential('upper_truncated_seps', normal_lccdf(sep_observed, asep_err, sep_ang_max))
        ##truncated_seps_lower = pm.Potential('upper_truncated_seps', normal_lccdf(sep_observed, asep_err, sep_ang_max))
        
        # contrast ratio limits - function of separations
        truncated_crs = pm.Potential('crs_truncated', normal_lccdf(cr_observed_inverse, cr_err_inverse, icr_max(sep_observed)))
        ##truncated_crs = pm.Potential('crs_truncated', normal_lccdf(cr_observed_inverse, cr_err_inverse, 1/7))

        
        
        # RUNNING THE FIT
        # -------------------
        traces = pm.sample(tune=nsteps, draws=nsteps, step=None, chains=1)
        
        # output as dataframe
        df = pm.trace_to_dataframe(traces)
    return df