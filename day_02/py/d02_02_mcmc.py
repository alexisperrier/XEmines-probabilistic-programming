'''
https://github.com/twiecki/WhileMyMCMCGentlySamples/blob/master/content/downloads/notebooks/MCMC-sampling-for-dummies.ipynb
'''

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


if __name__ == '__main__':

    # generate 20 samples from a Normal Distribution centered at 10
    # true_mu = 10
    data = np.random.randn(20) + 10
    sns.distplot(data)

    '''
        P(mu /D) = p( D/ mu  ) P(mu) / P(D)

    We want to find mu
    Assume data follows a N(mu, 1) with mu unknown

    => likelihood is also Normal distribution
    likelihood:  p( D/ mu ) ~ N(mu, 1)   # based on data

    We think a good way to represent the distribution of the unknown parameter mu is also N(0,1)
    prior:      p(mu) ~ N(0,1)           # our choice

    Since prior and posterior both ~N( , ) then posterior also follows a N( , ) distribution
    '''

    '''
    cte: p_w, prior ~ N(0,1), mu_0 =0
    MCMC: init, current vs proposal
    mu_c (= mu_0)
    mu proposal (new)
        mu_p = N(mu_c, p_w)  # Sample from Normal centered in mu_c with width: w_p
    Likelihood
        current L_c = Normal( mu_c, 1 ).pdf(data).prob()
        proposal L_p = Normal( mu_p, 1 ).pdf(data).prob()
    prior:
        current: p_mu_c = N(0,1).pdf(mu_c)
        proposal: p_mu_p = N(0,1).pdf(mu_p)

    ratio:
        p_accept = L_c * p_mu_c / (L_p * p_mu_p)
        if p_accept >> 1 => current is better => keep
        if p_accept << 1 => new is better => change

    '''

    # --------------------------------------------------------------------
    #  Version original
    # --------------------------------------------------------------------

    # generate data ~ N(10,2)
    true_mu, sigma = 10, 2
    data = np.random.randn(20) * sigma + true_mu

    # Initialize
    mu_current      = 0
    proposal_width  = 2.0           # SD of N(mu_current, proposal_width ) used to suggest mu_proposal
    trace           = [mu_current]  # Memorization of accepted values of mu
    rejected        = []
    ratio_list        = []
    suggested = []

    for i in range(1000):
        # consider new point: sample from of N(mu_c,p_w)
        mu_proposal = st.norm(mu_current, proposal_width).rvs()

        # Likelihood:
        likelihood_current  = np.log( norm(mu_current, 1.0).pdf(data) ).sum()
        likelihood_proposal = np.log( norm(mu_proposal, 1.0).pdf(data) ).sum()

        # Prior
        prior_current  = norm(np.mean(data), 1).pdf(mu_current)
        prior_proposal = norm(np.mean(data), 1).pdf(mu_proposal)

        # calculate Numerator of Bayes formula
        p_current  = likelihood_current * prior_current
        p_proposal = likelihood_proposal * prior_proposal

        ratio = p_current /  p_proposal
        accept = (np.random.rand()  + 1) > ratio
        ratio_list.append(ratio)
        if accept:
            trace.append(mu_current)
            mu_current = mu_proposal
        else:
            rejected.append(mu_proposal)



# -----------
