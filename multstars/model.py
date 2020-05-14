# imports
import numpy as np
import pandas as pd
import seaborn as sns
import pymc3 as pm
from pymc3.distributions.dist_math import normal_lcdf, normal_lccdf
import theano.tensor as tt

# survey-specific functions
from multstars.survey_transformations import imr_to_cr
from multstars.survey_limits import *


def pymc3_hrchl_fit(data, tune=1000, nsteps=1000, random_seed=0):
    
    asep = data['asep'].values
    asep_err = data['asep_err'].values
    cr = np.abs(data['cr'].values)
    cr_err = data['cr_err'].values
    parallax = data['parallax'].values


    with pm.Model() as hierarchical_model:

        # PRIORS
        # -------------
        
        # Separations 
        # (for normal disribution of log of separations)
        width = pm.Gamma('width', mu=1.1, sigma=0.5)
        center = pm.Gamma('center', mu=5, sigma=3)
        
        # power_index 
        power_index = pm.Normal('power_index', mu=1.2, sigma=.1)
        
        
        # MODELS OF POPULATIONS PHYSICAL PROPERTIES
        # --------------
        
        # Gaussian model for separations (in AU)
        sep_physical = pm.Lognormal('sep_physical', mu=center, sigma=width, shape=len(asep))
        
        # Mass Ratios - inverted
        mass_ratios_inverted = pm.Pareto('mass_ratios_inverted', alpha=power_index, m=1, shape=len(cr))
        
        
        # MAPPING FROM PHYSICAL TO OBSERVED PROPERTIES
        # ---------------

        #  physical separations to angular separations
        sep_angular = pm.Deterministic('sep_angular', sep_physical * parallax)
        
        # inverted mass ratios to inverted contrast ratios
        contrast_ratios = pm.Deterministic('contrast_ratios', imr_to_cr(mass_ratios_inverted))
        
        
        
        # LIKELIHOODS, WITH MEASUREMENT ERROR
        # -----------------
        
        # separations
        sep_observed = pm.Normal('sep_observed', mu=sep_angular, sigma=asep_err, observed=asep)
        
        # contrast ratios
        cr_observed = pm.Normal('cr_observed', mu=contrast_ratios, sigma=cr_err, observed=cr)
        
        
        # ACCOUNTING FOR OBSERVATION LIMITS
        # ------------------
        
        # integrate out the points which fall outside of the observation limits from the likelihood
        # pm.Potential accounts for truncation and normalizes likelihood
        
        # separation limits
        truncated_seps = pm.Potential(
            'truncated_seps',
            -pm.math.logdiffexp(normal_lcdf(sep_angular, asep_err, sep_max(cr)),
                                normal_lcdf(sep_angular, asep_err, sep_min(cr))))
        
        # contrast ratio limits
        truncated_crs = pm.Potential(
            'truncated_crs',
            -pm.math.logdiffexp(normal_lcdf(contrast_ratios, cr_err, cr_max(asep)),
                                normal_lcdf(contrast_ratios, cr_err, cr_min(asep))))

        
        
        # RUNNING THE FIT
        # -------------------
        traces = pm.sample(tune=tune, draws=nsteps, step=None, chains=1, random_seed=random_seed)
        
        # output as dataframe
        df = pm.trace_to_dataframe(traces)
    return traces, df