# AB testing

# http://localhost:8888/notebooks/Bayesian%20Methods%20for%20Hackers/Chapter2_MorePyMC/Ch2_MorePyMC_PyMC3.ipynb

import pandas as pd
import numpy as np
import pymc3 as pm
import seaborn as sns
import scipy.stats as stats


if __name__ == '__main__':

    '''
    one website, estimate conversion rate
    We assume all visitors have the same probability p of converting
    We model the conversion distribution as a Bernoulli(p)
    The data is composed of 0s and 1s (has not / has converted)
    We want to estimate the p.
    '''

    # Start by generating some data, 1500 samples / visits, probability of converting p = 0.05 (5% chance)
    p_true  = 0.05  # remember, this is unknown.
    N       = 1500
    occurrences = stats.bernoulli.rvs(p_true, size=N)

    # define the prior
    with pm.Model() as model:
        p = pm.Uniform('p', lower = 0, upper = 1)

    # now link the obesrved data to the prior distribution
    with model:
        obs = pm.Bernoulli("obs", p, observed=occurrences)
        step = pm.Metropolis()
        trace = pm.sample(20000, step=step)
        trace = trace[1000:]

    fig, ax = plt.subplots(1,1)
    plt.title("Posterior distribution of $p_A$, the true effectiveness of site A")
    sns.distplot(trace["p"])
    plt.vlines(p_true, 0, 90, linestyle="--", label="true $p_A$ (unknown)")
    plt.legend()
    plt.show()

    '''
    pm.summary(trace) shows a p centered on 0.041948 so different than the true_p
    but rhat and n_eff shows we have converged
    see also pm.traceplot(trace)

    What happens if we have less samples, or more samples?

    '''


    # --------------------------------------------------------------------
    # two websites
    # --------------------------------------------------------------------
    '''
    Now we test how 2 websites A and B perform (or two landing pages, ad campaigns, etc ....)
    '''
    true_p_A = 0.05
    true_p_B = 0.03

    # notice the unequal sample sizes -- no problem in Bayesian analysis.
    N_A = 75
    N_B = 1500

    # generate some observations
    observations_A = stats.bernoulli.rvs(true_p_A, size=N_A)
    observations_B = stats.bernoulli.rvs(true_p_B, size=N_B)

    # Set up the pymc3 model. Again assume Uniform priors for p_A and p_B.
    '''
    We want to understand 2 things: the conversion rate of each website p_A and p_B
    and the probability that one is better than the other p( p_B >  p_A)
    so on top of defining a model for each website, we also define a new variable,
    the difference between p_B  and  p_A
    When sampling the posterior of p_B and  p_A we will also obtain the distribution of the difference.
    '''
    with pm.Model() as model:
        p_A = pm.Uniform("p_A", 0, 1)
        p_B = pm.Uniform("p_B", 0, 1)

        # Define the deterministic delta function.
        delta = pm.Deterministic("delta", p_A - p_B)

        # Set of observations, in this case we have two observation datasets.
        obs_A = pm.Bernoulli("obs_A", p_A, observed=observations_A)
        obs_B = pm.Bernoulli("obs_B", p_B, observed=observations_B)

        # To be explained in chapter 3.
        step = pm.Metropolis()
        trace = pm.sample(20000, step=step)

    # plot posteriors

    fig, axs = plt.subplots(1,1, figsize=(9, 6))
    plt.title("Posterior distributions of $p_A$, $p_B$, and delta unknowns")
    sns.distplot(trace["p_A"], label="posterior of $p_A$")
    plt.vlines(true_p_A, 0, 80, linestyle="--", label="true $p_A$ (unknown)")

    sns.distplot(trace["p_B"], label="posterior of $p_B$")
    plt.vlines(true_p_B, 0, 80, linestyle="-.", label="true $p_B$ (unknown)")

    sns.distplot(trace["delta"], label="posterior of $\delta$")
    plt.vlines(true_p_A - true_p_B, 0, 80, label="true $\delta$ (unknown)")
    plt.tight_layout()

    plt.legend()

    # majority of Delta > 0 => string probability P_A > P_B

    print("Probability site A is WORSE than site B: %.3f" % \
        np.mean(trace["delta"] < 0))

    print("Probability site A is BETTER than site B: %.3f" % \
        np.mean(trace["delta"] > 0))


    # ---------------------------------------------------------------
    #  Beta - hierarchical
    # ---------------------------------------------------------------
    '''
    Let's now say we have several campaigns running at different times with different volumes.
    The above method does not scale
    We're going hierarchical
    We assume
    - We have Q campaigns running at different volumes and times
    - Different conversion rates for each campaign but all the ps (theta) are taken from the same distribution.
    - That distribution is a pm.Beta(a,b) distribution which is appropriate to model N trials with K success
    In that case a = N+1, b = K+1
    - Our observed data is taken from a Binomial distribution, with N trials and success probability theta
    - Theta is of size Q (number of campaigns). Each campaign has a different conversion rate
    - Furthermore we assume that a,b are constrained.
    '''

    # ---------------------------------------------------------------
    #  Double Click data? different campaigns
    # ---------------------------------------------------------------
    '''
    We have 132 days of campaigns geotargetd at different locations
    For each campaign we have the total number of impressions and the number of clicks,
    => ctr.

    Our goal now is to understand which geolocation is best

    Start with simple data exploration
    Then we'll model the AB testing
    '''


    df = pd.read_csv('digital_campaign.csv')

    # ----------------------------------------------------------------------
    # Comparing geolocs
    df = df.groupby(by = 'geoloc').sum().reset_index()
    df['ctr'] = df.clicks / df.impressions
    # ----------------------------------------------------------------------

    # 1. no constrain on the prior

    with pm.Model() as model:
        # a et b ?
        a = pm.Uniform('a', 0,100)
        b = pm.Uniform('b', 0,100)

        theta   = pm.Beta('theta', a, b, shape = df.shape[0])
        posterior     = pm.Binomial('posterior', p=theta, n=df.impressions, observed=df.clicks)
        trace   = pm.sample(100000, step = pm.Metropolis() )

    pm.summary(trace)

    '''
    What type of Beta distribution did we obtain?
    Does it make sense?
    Did the sampling converge ?
    Are the conversion rates trustworthy?
    '''
    # a and b need to be constrained
    #  this as analogous to the regularization parameter in a Lasso or Ridge regression
    # see https://stats.stackexchange.com/questions/94303/hierarchical-bayesian-modeling-of-incidence-rates/94310#94310
    # https://stackoverflow.com/questions/23198247/custom-priors-in-pymc
    with pm.Model() as model:
        ratio = pm.Uniform('ratio', 0, 1)
        inv_root_sample = pm.Uniform('inv_root_sample', 0, 1)
        a = pm.Deterministic('a', (ratio/inv_root_sample**2))
        b = pm.Deterministic('b', (1 - ratio)/(inv_root_sample**2))
        theta   = pm.Beta('theta', a, b, shape = df.shape[0])
        posterior     = pm.Binomial('posterior', p=theta, n=df.impressions, observed=df.clicks)
        trace   = pm.sample(100000, step = pm.Metropolis() )


    # another type of normalization

    # example 2
    import theano.tensor as tt

    def logp_ab(value):
        return tt.log(tt.pow(tt.sum(value), -5/2))

    with pm.Model() as model:
        ab = pm.HalfFlat('ab', shape=2, testval=np.asarray([1., 1.]))
        pm.Potential('p(a, b)', logp_ab(ab))

        alpha = pm.Deterministic('alpha', ab[0])
        beta = pm.Deterministic('beta', ab[1])
        theta = pm.Beta('theta', alpha=ab[0], beta=ab[1], shape=df[0:2].shape[0])
        p = pm.Binomial('y', p=theta, n=df.impressions[0:2].values, observed=df.clicks[0:2].values)
        trace = pm.sample(100000, step = pm.Metropolis())


    # another example
    # https://dsaber.com/2016/08/27/analyze-your-experiment-with-a-multilevel-logistic-regression-using-pymc3%E2%80%8B/

    with pm.Model() as bb_model:

        def ab_likelihood(value):
            a = value[0]
            b = value[1]

            return T.switch(T.or_(T.le(a, 0), T.le(b, 0)),
                            -np.inf,
                            np.log(np.power((a + b), -2.5)))

        ab = pm.DensityDist('ab', ab_likelihood, shape=2, testval=[1.0, 1.0])

        a = ab[0]
        b = ab[1]

        rates = pm.Beta('rates', a, b, shape=4)

        trials      = np.array([n] * 4)
        successes   = np.array(ss)

        obs = pm.Binomial('observed_values', trials, rates,
                          observed=successes)



# --------        #
