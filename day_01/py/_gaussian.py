'''
# Gaussian mixture
1. gaussian distribution
2. estimate mu sigma

# Resources
- http://people.duke.edu/~ccc14/sta-663-bootstrap/PyMC3.html#estimating-mean-and-standard-deviation-of-normal-distribution
- https://brilliant.org/wiki/gaussian-mixture-model/
Definition, EM method, unsupervised approach, equations
- https://scikit-learn.org/stable/modules/mixture.html
EM based and variational inference based


case: 2 gaussians, not same number of samples, estimate mu and sigma with pymc3
[notebook](https://docs.pymc.io/notebooks/marginalized_gaussian_mixture_model.html)

https://docs.pymc.io/notebooks/marginalized_gaussian_mixture_model.html
https://docs.pymc.io/notebooks/gaussian_mixture_model.html



'''

import pandas as pd
import numpy as np
import pymc3 as pm
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # ----------------------------------------------------------------------
    #  1. generating a gaussian
    # ----------------------------------------------------------------------
    N       = 1000
    mu_      = 10
    sigma_   = 2
    y = np.random.normal(mu_, sigma_, N)
    fig, ax = plt.subplots()
    sns.distplot(y,  bw=1)
    plt.vlines()
    plt.show()
    # ----------------------------------------------------------------------
    #  2. estimate the mean and std
    # http://people.duke.edu/~ccc14/sta-663-bootstrap/PyMC3.html#estimating-mean-and-standard-deviation-of-normal-distribution
    # ----------------------------------------------------------------------
    n_iteration = 10000

    with pm.Model() as model:
        # define priors (low information, assume little)
        mu      = pm.Uniform('mu', lower=0, upper=100)
        sigma   = pm.Uniform('sigma', lower=0, upper=10)

        # define likelihood
        y_obs   = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=y)

        # inference
        trace   = pm.sample(n_iteration, pm.Metropolis(), pm.find_MAP(), progressbar=True)


    # the trace object
    print("trace.varnames {}".format(trace.varnames))
    # print("trace.stat_names {}".format(trace.stat_names))

    print(" Infered mu {:.2f} np.mean  {:.2f}".format( np.mean(  trace['mu'][1000:]), np.mean(y) ))
    print(" Infered sigma {:.2f} np.std  {:.2f}".format(np.mean(trace['sigma'][1000:]) , np.std(y) ))

    print(pm.summary(trace))

    # viz
    fig, ax = plt.subplots(1,2, figsize = (10,5))
    plt.subplot(2,1,1)
    sns.distplot(trace['mu'][1000:])
    plt.vlines(mu_, 0, 6, label="real mu")
    plt.vlines(np.mean(y), 0, 6, color = 'red', label="infered mu")
    plt.title("mu")
    plt.legend()
    plt.subplot(2,1,2)
    sns.distplot(trace['sigma'][1000:])
    plt.vlines(sigma_, 0, 6, label="real mu")
    plt.vlines(np.std(y), 0, 6, color = 'red', label="infered mu")
    plt.title("sigma")
    plt.legend()
    plt.tight_layout()
    plt.show()


    fig, ax = plt.subplots()
    pm.forestplot(trace)
    fig, ax = plt.subplots()
    pm.traceplot(trace)
    pm.summary(trace)




# ---------------------------
