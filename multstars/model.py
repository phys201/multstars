# imports
import numpy as np
import pandas as pd
import seaborn as sns
import pymc3 as pm
from pymc3.distributions.dist_math import normal_lcdf, normal_lccdf
import theano.tensor as tt



# LIMITS FOR WHAT WILL APPEAR IN THE FINAL DATA

# limits for separations
sep_ang_max = 4.0


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


def pymc3_hrchl_fit(data, nsteps=1000):
    
    asep = data['asep'].values
    asep_err = data['asep_err'].values
    cr = np.abs(data['cr'].values)
    cr_err = data['cr_err'].values
    parallax = data['parallax'].values
    
    cr_inverse = 1 / cr
    cr_err_inverse = 1 / cr_err


    with pm.Model() as hierarchical_model:

        # PRIORS
        # -------------
        
        # Separations 
        # (for normal disribution of log of separations)
        width = pm.Gamma('width', mu=1.1, sigma=0.5)
        center = pm.Gamma('center', mu=5, sigma=3)
        
        # power_index 
        power_index = pm.Normal('power_index', mu=1.2, sigma=.2)
        
        
        # MODELS OF POPULATIONS PHYSICAL PROPERTIES
        # --------------
        
        # Gaussian model for separations (in AU)
        sep_physical = pm.Lognormal('sep_physical', mu=center, sigma=width)
        
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
        sep_observed = pm.Normal('sep_observed', mu=sep_angular, sigma=asep_err, observed=asep)
        
        # contrast ratios
        cr_observed_inverse = pm.Normal('cr_observed', mu=contrast_ratios_inverted, sigma=cr_err_inverse, observed=cr_inverse)
        
        
        # ACCOUNTING FOR OBSERVATION LIMITS
        # ------------------
        
        # integrate out the points which fall outside of the observation limits from the likelihood
        # pm.Potential accounts for truncation and normalizes likelihood
        
        # separation limits
        truncated_seps_upper = pm.Potential('upper_truncated_seps', normal_lccdf(sep_observed, asep_err, sep_ang_max))
        truncated_seps_lower = pm.Potential('lower_truncated_seps', normal_lcdf(sep_observed, asep_err, sep_min(cr_observed_inverse)))
        
        # contrast ratio limits - function of separations
        truncated_crs = pm.Potential('crs_truncated', normal_lccdf(cr_observed_inverse, cr_err_inverse, icr_max(sep_observed)))

        
        
        # RUNNING THE FIT
        # -------------------
        traces = pm.sample(tune=nsteps, draws=nsteps, step=None, chains=1)
        
        # output as dataframe
        df = pm.trace_to_dataframe(traces)
    return df