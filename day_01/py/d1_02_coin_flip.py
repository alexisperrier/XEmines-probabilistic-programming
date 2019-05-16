'''
Demo coin flip
In this notebook we modelize coin the ancient art of coin flipping and the question of the bias of said coin.
We start by showing the mirage of frequentists point estimates and demonstrate a more sane approach
with a simple Byesian decision process.
We move on to .... using better priors


# Coin flip analysis by MCMC, hand coded: convergence and moves
https://rstudio-pubs-static.s3.amazonaws.com/387431_25b97abbf9c040af9891d29144c6f176.html

# PyMC Tutorial #1: Bayesian Parameter Estimation for Bernoulli Distribution
https://alfan-farizki.blogspot.com/2015/07/pymc-tutorial-bayesian-parameter.html
in length post looking at EM and PP methods.

# Bayesian Inference of a Binomial Proportion - The Analytical Approach
https://www.quantstart.com/articles/Bayesian-Inference-of-a-Binomial-Proportion-The-Analytical-Approach
All about Binomial, and beta distributions

# Coin toss
http://people.duke.edu/~ccc14/sta-663-bootstrap/PyMC3.html#coin-toss

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
    '''
    Let's start by generating some data of a coin flip with p(Coin = 0) = 0.2
    and calculating the Maximum Likelihood Estimate
    '''

    n  = 30
    p  = 0.2

    data     = st.bernoulli.rvs(p, size=n)
    p_mle    = np.sum(data) / n # np.mean(data)
    print("MLE estimation of p:  {:.2f}".format(p_mle))

    '''
    The point estimate is not even close to the real value, but we had only a few samples.
    Let's see how the point estimate varies with the number of samples
    '''
    # -------------------------------------------------------------
    #  MLE for different data size
    # -------------------------------------------------------------
    p  = 0.4

    p_hat = []
    for n in range(100,10000,100):
        data     = st.bernoulli.rvs(p, size=n)
        p_mle    = np.sum(data) / n
        p_hat.append(np.mean(data))
        # print("[{}]  p ~=  {:.4f} {:.4f} ".format(n, p_mle, np.mean(data)))

    fig,ax = plt.subplots(1,1)
    plt.plot( range(500,10000,100), p_hat[4:]  )
    plt.hlines(p, 500, 10000)
    plt.show()

    '''
    Still lots of variations around the true value
    We don't see a clear convergence, after a certain number of samples
    With frequentist: There's no way to use our knowledge
    that usually coins are fair and that p ~=0.5
    '''

    # -------------------------------------------------------------
    # Same thing with PyMC3
    # -------------------------------------------------------------
    '''
    So let's infer the coin bias with PyMC3.

    The distribution of the random variable: number of success over N trials
    (count number of heads after N coin flips with p = p(coin == Head) )
    is the Binomial distribution.
    '''

    x = np.arange(0, 40)
    ns = [10, 15, 20]
    ps = [0.5, 0.5, 0.5]
    for n, p in zip(ns, ps):
        pmf = st.binom.pmf(x, n, p)
        plt.plot(x, pmf, '-o', label='n = {}, p = {}'.format(n, p))
    plt.xlabel('x', fontsize=14)
    plt.ylabel('f(x)', fontsize=14)
    plt.legend(loc=1)
    plt.show()


    '''
    How to use scipy.stats to generate data and pmf
    - data:
        st.<distribution>.rvs
    - distribution Probability mass function: probability RV == value
        st.<distribution>.rvs

    The probability of getting exactly k successes in n trials is given by the probability mass function:
    {\displaystyle f(k,n,p)=\Pr(k;n,p)=\Pr(X=k)={\binom {n}{k}}p^{k}(1-p)^{n-k}}

    For instance, for p = 0.3, probability of getting 6 Heads in a row is
    f(6,6,0.3) = (6! / 6! 0! ) ( 0.3 )^6 (1-0.3)^(6-6) = 0.3^6 = 0.000729

    '''

    '''
    Let's generate some data first (p, n),
    '''

    number_trials = 100
    p       = 0.4
    data    = st.bernoulli.rvs(p, size=number_trials)

    '''
    - [prior] No assumptions of p: Uniform over [0,1]
    - [Likelihood] as Binomial
    - Sampler: classic Metropolis
    '''

    with pm.Model() as model:
        # Prior, no assumption at all P(θ)
        p       = pm.Uniform('p', 0,1)
        # Likelihood P(x1,x2,...,xn|θ)
        y       = pm.Binomial('y', n = number_trials, p=p, observed=data)
        # Sample posterior distribution
        # P(θ|X)
        step    = pm.Metropolis()
        trace   = pm.sample(20000, step = step)

    print(pm.summary(trace))
    print("\n-- Estimated mean for p: {:.4f}".format(np.mean(trace['p']) * number_trials) )

    pm.traceplot(trace)

    pm.forestplot(trace)


    '''
    Visualize the model graph with graphviz
        conda install -c conda-forge python-graphviz
    '''
    pm.model_to_graphviz(model)

    '''
    - increase number of trials
    The variance of the trace is reduced: we have more trust in our predictions

    - plot trace, trace summary
    - ! mean(p) is divided by number of trials
    '''

    '''
    No change Prior to beta(a,b)
    Because Beta is best distribution to model coin flip
    and Beta is prior conjugate of Bernoulli

        p∼Beta(α,β)
        y∼Binomial(n,p)
    '''

    fig, ax = plt.subplots(1,1)
    x = np.linspace(0, 1, 200)
    alphas = [0.5, 0.5, 0.5, 2, 2, 2]
    betas = [0.5, 1, 2, 0.5, 1, 2]
    for a, b in zip(alphas, betas):
        pdf = st.beta.pdf(x, a, b)
        plt.plot(x, pdf, label=r'$\alpha$ = {}, $\beta$ = {}'.format(a, b))
    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.ylim(0, 4.5)
    plt.legend(loc=9)
    plt.show()


    '''
    Let's take beta(2,2) as prior for p
    '''


    number_trials = 100
    data     = st.bernoulli.rvs(0.4, size=number_trials)
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
    pm.autocorrplot(trace)
    pm.energyplot(trace)

    # see also
    plt.hist(trace['p'], 15, histtype='step',  label='post');

    '''
    But we chose a,b quite arbitrarily. We could also infer them from the data
    Assume a,b come from a Uniform distribution over 0, 10
    '''
    number_trials = 10000
    data     = st.bernoulli.rvs(0.4, size=number_trials)
    with pm.Model() as model:
        a       = pm.Uniform('a', 0,10)
        b       = pm.Uniform('b', 0,10)
        p       = pm.Beta('p', a,b)
        y       = pm.Binomial('y', n = number_trials, p=p, observed=data)
        step    = pm.Metropolis()
        trace   = pm.sample(100000, step = step, progressbar=True)

    print(pm.summary(trace))
    print("\n-- Estimated mean for p: {:.4f}".format(np.mean(trace['p']) * number_trials) )
    pm.traceplot(trace)
    pm.plot_posterior(trace)


    # -------------------------------------------------------------
    # now with a dice, switch to categorical distribution
    # -------------------------------------------------------------

    n  = 100
    probas  = np.ones(6) / 6
    probas  = [1/7.]*5 + [2/7.]
    data    = np.random.multinomial(n, probas)

    with pm.Model() as model:
        probs = pm.Dirichlet('probs', a=np.ones(6))  # flat prior
        rolls = pm.Multinomial('rolls', n=100, p=probs, observed=data)
        trace = pm.sample(draws= 1, tune=500, discard_tuned_samples=False)

    trace['probs']  # posterior samples of how fair the dice are




# ----------------------------
