# imports
import numpy as np
import pandas as pd
import seaborn as sns
import emcee

def likelihood(data, parameters):
    '''
    INPUT
    ------
    data: pandas data frame, containing physical separations and errors [AU] for binary systems
    parameters: tuple containing the center and width of a gaussian to fit the distribution of physical separations
    OUTPUT
    ------
    likelihood: value of the likelihood function
    '''
    mean, sigma = parameters
    p_sep = data['SEP_PHYSICAL']
    p_sep_e = data['E_SEP_PHYSICAL']
    
    # gaussian
    likelihoods = np.exp(-(p_sep-mean)**2 / (2*(sigma**2 + p_sep_e**2))) / (np.sqrt(2*np.pi) * (sigma**2 + p_sep_e**2))
    return likelihoods.sum()

def log_prior(parameters, p_sep, p_sep_e):
    '''
    INPUT
    ------
    data: pandas data frame, containing physical separations and errors [AU] for binary systems
    parameters: tuple containing the center and width of a gaussian to fit the distribution of physical separations
    OUTPUT
    ------
    prior: value of the prior
    '''
    
    mean, sigma = parameters
    
    # setting ranges
    mean_min = 0.01 ; mean_max = 100
    sigma_min = 0.01 ; sigma_max = 200

    mean_p = 1.0 / (mean_max - mean_min)                  # flat prior
    
    sigma_p = 1 / (sigma * np.log(sigma_max/sigma_min))   # jeffrey's prior
    
    # if prameters within range...    
    if (mean_min<mean<mean_max) and (sigma_min<sigma<sigma_max):
        return mean_p*sigma_p
    return -np.inf


def log_likelihood(parameters, p_sep, p_sep_e):
    mean, sigma = parameters
    # gaussian
    likelihoods = np.exp(-(p_sep-mean)**2 / (2*(sigma**2 + p_sep_e**2))) / (np.sqrt(2*np.pi) * (sigma**2 + p_sep_e**2))
    return np.log(likelihoods.sum())
    
def log_posterior(p_sep, p_sep_e, parameters):
    '''
    INPUT
    ------
    data: pandas data frame, containing physical separations and errors [AU] for binary systems
    parameters: tuple containing the center and width of a gaussian to fit the distribution of physical separations
    OUTPUT
    ------
    prior: value of the posterior
    '''
    return log_prior(p_sep, p_sep_e, parameters) + log_likelihood(p_sep, p_sep_e, parameters)


# based on week 9 section notebook
def emcee_fit(data, map_est, nwalkers=50, nsteps=100, metropolis=False):
    p_sep = data['SEP_PHYSICAL']
    p_sep_e = data['E_SEP_PHYSICAL']
    gaussian_ball_width = 1e-4
    ndim = len(map_est)
    
    # For simple Metropolis-Hastings sampling, pass metropolis=True
    
    gaussian_ball = gaussian_ball_width * np.random.randn(nwalkers, ndim)
    starting_positions = (1 + gaussian_ball) * map_est

    
    if metropolis:
        sampler = emcee.MHSampler
        proposal_covariance_matrix = np.diag(map_est) / 100
        first_argument = proposal_covariance_matrix
        starting_positions = starting_positions[0]
        nwalkers = 1
    else:
        sampler = emcee.EnsembleSampler
        first_argument = nwalkers
    sampler = sampler(first_argument, ndim, log_posterior, args=(p_sep, p_sep_e))
    sampler.run_mcmc(starting_positions, nsteps)
    
    df = pd.DataFrame(np.vstack(sampler.chain))
    df.index = pd.MultiIndex.from_product([range(nwalkers), range(nsteps)], names=['walker', 'step'])
    df.columns = ['mean','width']
    return df