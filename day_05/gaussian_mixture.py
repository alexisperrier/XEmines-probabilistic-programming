'''
# Gaussian mixture
1. gaussian distribution
2. estimate mu sigma

3. 2 gaussians
4. infer the parameters of the 2 gaussians

5. multivariate GM
6. infer weights


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

How to build a Gaussian Mixture Model
https://hub.packtpub.com/how-to-build-a-gaussian-mixture-model/


Scikit
'''

import pandas as pd
import numpy as np
import pymc3 as pm
import seaborn as sns
import matplotlib.pyplot as plt
# sns.jointplot(x="x", y="y", data=df, kind="kde");

if __name__ == "__main__":

    # ----------------------------------------------------------------------
    #  1. generating a gaussian, then a couple of gaussians
    # ----------------------------------------------------------------------
    N       = 1000
    mu_      = 10
    sigma_   = 2
    y = np.random.normal(mu_, sigma_, N)
    fig, ax = plt.subplots()
    sns.kdeplot(y,  bw=1)
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

    '''  Ex.1
            - increase number of samples available
            - increase n_iteration
            impact on traceplot ? estimation ? confidence intervals ?
            Rhat?
    '''

    # ----------------------------------------------------------------------
    #  3. generate multi GMM
    #  https://hub.packtpub.com/how-to-build-a-gaussian-mixture-model/
    # ----------------------------------------------------------------------
    # Number of Gaussians
    clusters = 2
    # Number of points in each Gaussian
    n_cluster   = [100,500]
    # means and st of each Gaussian
    means       = [3, 22]
    std_devs    = [2, 5]

    mix = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster))

    # sns.kdeplot(mix, bw = 0.5)
    plt.plot(mix, '.')
    plt.plot(mix[:n_cluster[0]], 'r.')


    # ----------------------------------------------------------------------
    #  4. find which Gaussian the point belongs to, knowing the gaussians
    # logistic regression
    # ----------------------------------------------------------------------
    '''
    This does not work. Need a 2 stage
    '''
    with pm.Model() as model:

        alpha = pm.Normal('alpha', mu = 3, sd = 2)
        beta  = pm.Normal('beta', mu = 22, sd = 5)
        gmm   = pm.Potential('gmm', alpha + beta)


        trace = pm.sample(10000)

    pm.traceplot(trace)







    # ----------------------------------------------------------------------
    #  5. find the parameters of the gaussians
    # ----------------------------------------------------------------------

    with pm.Model() as model:

        # Dirichlet log likelihood
        # generalisation of the beta distribution from 2 to K
        p           = pm.Dirichlet('p', a=[1,1])
        category    = pm.Categorical('category', p=p, shape=n_total)

        means       = pm.math.constant(means)

        y           = pm.Normal('y', mu=means[category], sd=2, observed=mix)

        step1       = pm.ElemwiseCategorical(vars=[category], values=range(clusters))
        step2       = pm.Metropolis(vars=[p])

        trace_kg = pm.sample(10000, step=[step1, step2])


    pm.traceplot(trace_kg)


    # ----------------------------------------------------------------
    #  ????
    # ----------------------------------------------------------------

    k = 3
    ndata = 500
    spread = 5
    centers = np.array([-spread, 0, spread])

    # simulate data from mixture distribution
    v = np.random.randint(0, k, ndata)
    data = centers[v] + np.random.randn(ndata)

    plt.hist(data);

    with pm.Model() as model:
        # cluster sizes
        p = pm.Dirichlet('p', a=np.array([1., 1., 1.]), shape=k)
        # cluster centers
        means = pm.Normal('means', mu=[0, 0, 0], sd=15, shape=k)

        # measurement error
        sd = pm.Uniform('sd', lower=0, upper=20)

        # latent cluster of each observation
        category = pm.Categorical('category', p=p, shape=ndata)

        # likelihood for each observed value
        points = pm.Normal('obs',
                           mu=means[category],
                           sd=sd,
                           observed=data)

        step1 = pm.Metropolis(vars=[p, sd, means])
        step2 = pm.ElemwiseCategorical(vars=[category], values=[0, 1, 2])
        step2 = pm.CategoricalGibbsMetropolis(vars=[category], values=[0, 1, 2])

        trace = pm.sample(10000, step=[step1, step2])


pm.summary(trace)


    # ----------------------------------------------------------------
    #  ????
    # ----------------------------------------------------------------

    k = 2
    ndata = 500
    spread = 5
    centers = np.array([-spread, spread])

    # simulate data from mixture distribution
    v = np.random.randint(0, k, ndata)
    data = centers[v] + np.random.randn(ndata)

    # plt.hist(data, bins = 50);

    with pm.Model() as model:
        # cluster sizes
        p = pm.Dirichlet('p', a=np.array([1., 1.]), shape=k)
        # cluster centers
        means = pm.Normal('means', mu=[0, 0], sd=15, shape=k)

        # measurement error
        sd = pm.Uniform('sd', lower=0, upper=20)

        # latent cluster of each observation
        category = pm.Categorical('category', p=p, shape=ndata)

        # likelihood for each observed value
        points = pm.Normal('obs',
                           mu=means[category],
                           sd=sd,
                           observed=data)


    with pm.Model() as model:
        step1 = pm.Metropolis(vars=[p, sd, means])
        step2 = pm.ElemwiseCategorical(vars=[category], values=[0, 1])
        step2 = pm.CategoricalGibbsMetropolis(vars=[category], values=[0, 1])

        trace = pm.sample(10000, step=[step1, step2])


pm.summary(trace)
pm.traceplot(trace[500:], ['p', 'sd', 'means']);


    # ---------------------------------------------------------------------
    #  This works but does not converge
    # see https://stackoverflow.com/questions/21005541/converting-a-mixture-of-gaussians-to-pymc3
    # ---------------------------------------------------------------------

    n1 = 500
    n2 = 200
    n = n1+n2

    mean1 = 21.8
    mean2 = 42.0
    precision = 0.1

    sigma = np.sqrt(1 / precision)

    data1 = np.random.normal(mean1,sigma,n1)
    data2 = np.random.normal(mean2,sigma,n2)

    data = np.concatenate([data1 , data2])
    #np.random.shuffle(data)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, xlabel='x', ylabel='y', title='mixture of 2    guassians')
    ax.plot(range(0,n1+n2), data, 'x', label='data')
    plt.legend(loc=0)

    with pm.Model() as model:
        #priors
        p       = pm.Uniform( "p", 0 , 1) #this is the fraction that come from mean1 vs mean2

        ber     = pm.Bernoulli( "ber", p = p, shape=len(data)) # produces 1 with proportion p.

        sigma       = pm.Uniform('sigma', 0, 100)
        precision   = sigma**-2

        mean    = pm.Uniform('mean', 15, 60, shape=2)

        mu      = pm.Deterministic('mu', mean[ber])

        process = pm.Normal('process', mu=mu, tau=precision, observed=data)

    with model:
        step1 = pm.Metropolis([p, sigma, mean])
        step2 = pm.ElemwiseCategorical(vars=[ber], values=[0, 1])
        trace = pm.sample(10000, [step1, step2])


 # see rhat > 1.4
 pm.summary(trace, ['p', 'sigma', 'mean'])
 # but traceplot is nice
 pm.traceplot(trace[500:], ['p', 'sigma', 'mean']);



# ----------------
