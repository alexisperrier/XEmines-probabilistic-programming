'''
Christopher Fonnesbeck Probabilistic Programming with PyMC3 PyCon 2017
Hierarchical linear regression
GLM
https://www.youtube.com/watch?v=5TyvJ6jXHYE
https://github.com/pymc-devs/pymc3/blob/master/pymc3/examples/data/radon.csv
https://docs.pymc.io/notebooks/GLM-hierarchical.html
https://nbviewer.jupyter.org/github/fonnesbeck/multilevel_modeling/blob/master/multilevel_modeling.ipynb?create=1
https://github.com/fonnesbeck/multilevel_modeling/blob/master/multilevel_modeling.ipynb
'''

import pandas as pd
import pymc3 as pm
import numpy as np
import theano

import matplotlib.pyplot as plt

if __name__ == "__main__":

    # data = pd.read_csv(pm.get_data('../data/radon.csv'))
    data = pd.read_csv('../data/radon.csv')
    # data['log_radon'] = data['log_radon'].astype(theano.config.floatX)
    county_names = data.county.unique()
    county_idx   = data.county_code.values
    n_counties   = len(data.county.unique())

    # Pooled
    # 𝑟𝑎𝑑𝑜𝑛𝑖,𝑐=𝛼+𝛽∗floor𝑖,𝑐+𝜖
    with pm.Model() as pooled_model:
        a   = pm.Normal('a', 0, sd = 100)
        b   = pm.Normal('b', 0, sd = 100)
        eps = pm.HalfCauchy('eps', 5)
        radon_est = a + b * data.floor.values
        # likelihood
        y = pm.Normal('y', radon_est, sd = eps, observed = data.log_radon.values )

    with pooled_model:
        pooled_trace = pm.sample(5000)

    pm.traceplot(pooled_trace)
    print(pm.summary(pooled_trace))


    # unpooled
    # 𝑟𝑎𝑑𝑜𝑛𝑖,𝑐=𝛼𝑐+𝛽𝑐∗floor𝑖,𝑐+𝜖𝑐
    with pm.Model() as unpooled_model:
        a   = pm.Normal('a', 0, sd = 100, shape = n_counties)
        b   = pm.Normal('b', 0, sd = 100, shape = n_counties)
        eps = pm.HalfCauchy('eps', 5)
        Cauradon_est = a[county_idx] + b[county_idx] * data.floor.values
        # likelihood
        y = pm.Normal('y', radon_est, sd = eps, observed = data.log_radon.values )

    with unpooled_model:
        unpooled_trace = pm.sample(5000)

    # pm.traceplot(unpooled_trace)
    # print(pm.summary(unpooled_trace))

    # Hierachical
    with pm.Model() as hierarchical_model:
        mu_a = pm.Normal('mu_a', 0, sd = 100)
        mu_b = pm.Normal('mu_b', 0, sd = 100)
        sigma_a = pm.HalfCauchy('sigma_a', 5)
        sigma_b = pm.HalfCauchy('sigma_b', 5)

        a   = pm.Normal('a', mu = mu_a, sd = sigma_a, shape = n_counties)
        b   = pm.Normal('b', mu = mu_b, sd = sigma_a, shape = n_counties)

        eps = pm.HalfCauchy('eps', 5)
        radon_like = a[county_idx] + b[county_idx] * data.floor.values
        # likelihood
        y = pm.Normal('y', radon_like, sd = eps, observed = data.log_radon.values )

    with hierarchical_model:
        hierarchical_trace = pm.sample(draws=2000, n_init=1000)


    pm.summary(hierarchical_trace)
    pm.traceplot(hierarchical_trace)


# ---------------
