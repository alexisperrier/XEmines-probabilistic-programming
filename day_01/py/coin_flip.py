'''
# Demo coin flip

https://rstudio-pubs-static.s3.amazonaws.com/387431_25b97abbf9c040af9891d29144c6f176.html
Coin flip analysis by MCMC: convergence and moves, no PyMC

PyMC Tutorial #1: Bayesian Parameter Estimation for Bernoulli Distribution
https://alfan-farizki.blogspot.com/2015/07/pymc-tutorial-bayesian-parameter.html
in length post looking at EM and PP methods.
'''
# ---------------------------------------------------------------------
# 1. Generate data using scipy.stats
# ---------------------------------------------------------------------

import numpy as np

import scipy.stats as st

import numpy as np
import pymc3 as pm
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":

    n  = 30
    p  = 0.2

    data     = st.bernoulli.rvs(p, size=size)
    p_mle    = np.sum(data) / n
    print("MLE estimation of p:  {:.2f}".format(p_mle))

    # -------------------------------------------------------------
    #  MLE for different data size
    # -------------------------------------------------------------
    for n in range(100,2000,100):
        data     = st.bernoulli.rvs(p, size=n)
        p_mle    = np.sum(data) / n
        print("[{}]  p ~=  {:.4f}".format(n, p_mle))

    # (still not converged with 2000 samples)

    # -------------------------------------------------------------
    # Same thing with PyMC3
    # -------------------------------------------------------------
    '''
    No assumption on p: uniform over 0,1
    - increase number of trials (is there an influence?)
    - try other sampler: NUTS
    - plot trace, trace summary
    - ! mean(p) is divided by number of trials
    '''
    data     = st.bernoulli.rvs(0.5, size=1000)
    number_trials = 100
    with pm.Model() as model:
        p       = pm.Uniform('p', 0,1)
        y       = pm.Binomial('y', n = number_trials, p=p, observed=data)
        step    = pm.Metropolis()
        trace   = pm.sample(20000, step = step, progressbar=True)

    print(pm.summary(trace))
    print("\n-- Estimated mean for p: {:.4f}".format(np.mean(trace['p']) * number_trials) )

    pm.traceplot(trace)

    # -------------------------------------------------------------
    # Change prior to Beta
    # see http://people.duke.edu/~ccc14/sta-663-bootstrap/PyMC3.html#coin-toss
    # What happens when beta coef are changed to 0.5, 0.5 instead of 2,2
    # n_eff drops, why?
    # Try other distributions for the prior: Normal, ...

    # -------------------------------------------------------------
    '''
    https://calogica.com/python/pymc3/2018/11/27/dice-polls-dirichlet-multinomial.html
    Thus, the probability p of a coin landing on head is modeled to be Beta distributed
    (with parameters α and β), while the likelihood of heads and tails is assumed
    to follow a Binomial distribution with parameters n (representing the number of flips)
    and the Beta-distributed p, thus creating the link.

    p∼Beta(α,β)
    y∼Binomial(n,p)
    '''

    data     = st.bernoulli.rvs(0.5, size=1000)
    number_trials = 100
    with pm.Model() as model:
        p       = pm.Beta('p', 2,2)
        y       = pm.Binomial('y', n = number_trials, p=p, observed=data)
        step    = pm.Metropolis()
        trace   = pm.sample(20000, step = step, progressbar=True)

    print(pm.summary(trace))
    print("\n-- Estimated mean for p: {:.4f}".format(np.mean(trace['p']) * number_trials) )

    # other plots
    pm.plot_posterior(trace)
    pm.traceplot(trace)
    pm.forestplot(trace)
    pm.autoorrplot(trace)
    pm.energyplot(trace)

    # see also
    plt.hist(trace['p'], 15, histtype='step',  label='post');

    # -------------------------------------------------------------
    # now with a dice, switch to categorical distribution
    # -------------------------------------------------------------

    n  = 100
    probas  = np.ones(6) / 6
    probas  = [1/7.]*5 + [2/7.]
    data     = np.random.multinomial(n, probas)

    with pm.Model() as model:
        probs = pm.Dirichlet('probs', a=np.ones(6))  # flat prior
        rolls = pm.Multinomial('rolls', n=100, p=probs, observed=data)
        trace = pm.sample(draws= 1, tune=500, discard_tuned_samples=False)

    trace['probs']  # posterior samples of how fair the die are




# ----------------------------
