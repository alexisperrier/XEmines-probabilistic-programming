'''
https://nbviewer.jupyter.org/github/fonnesbeck/multilevel_modeling/blob/master/multilevel_modeling.ipynb
'''

import sys

# scientific packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize

import pymc3 as pm
import patsy as pt

if __name__ == "__main__":

    df = pd.read_csv('../../data/radon.csv')

    df = df[['log_radon', 'floor']]
    with pm.Model() as model:
        pm.glm.GLM.from_formula('log_radon ~ floor', df)
        trace = pm.sample(10000, pm.NUTS())


    pm.traceplot(trace)
    pm.plot_posterior(trace)
    pm.summary(trace)


# ------------------------
    df = pd.read_csv('../../data/radon.csv')

    from statsmodels.formula.api import ols
    unpooled_fit = ols('log_radon ~ county + floor - 1', df).fit()
    unpooled_estimates = unpooled_fit.params


# ---------------------------

    counties = df.county.values
all_traces = pd.DataFrame()
for county in counties[0:10]:
    with pm.Model() as partial_pooling:
        # Priors
        mu_a    = pm.Normal('mu_a', mu=0., sd=1000)
        sigma_a = pm.Uniform('sigma_a', lower=0, upper=100)

        # Random intercepts
        radon_level_county = pm.Normal('radon_level_county', mu=mu_a, sd=sigma_a, observed=df[df.county == county].log_radon)

        # sigma_y = pm.Uniform('sigma_y', lower=0, upper=100)
        # y_like  = pm.Normal('y_like', mu=radon_level_county, sd=sigma_y, observed=df[df.county == county].log_radon)

        trace   = pm.sample(1000, pm.NUTS() )
        trace_df = pm.summary(trace)
        trace_df['county'] = county
        all_traces = pd.concat([all_traces, trace_df])
        print('--' * 20)
        print(county)
        print(trace_df)

# ------------------------------------
#  partial pooling
# ------------------------------------
with pm.Model() as varying_intercept:
    mu_a    = pm.Normal('mu_a', mu=0., tau=0.0001)
    sigma_a = pm.Uniform('sigma_a', lower=0, upper=100)


    # Random intercepts and common slope
    a = Normal('a', mu=mu_a, tau=tau_a, shape=len(set(counties)))
    b = Normal('b', mu=0., tau=0.0001)

    # Model error
    sigma_y = Uniform('sigma_y', lower=0, upper=100)

    # Expected value
    y_hat = a + b * list(df.floor)

    # Data likelihood
    y_like = Normal('y_like', mu=y_hat, tau=tau_y, observed=log_radon)


# --------------------------

county_names = df.county.unique()
county_idx = df['county_code'].values
with pm.Model() as hierarchical_model:
    # Hyperpriors
    mu_a    = pm.Normal('mu_alpha', mu=0., sd=1)
    sigma_a = pm.HalfCauchy('sigma_alpha', beta=1)
    mu_b    = pm.Normal('mu_beta', mu=0., sd=1)
    sigma_b = pm.HalfCauchy('sigma_beta', beta=1)

    # Intercept for each county, distributed around group mean mu_a
    a = pm.Normal('alpha', mu=mu_a, sd=sigma_a, shape=len(df.county.unique()))
    # Intercept for each county, distributed around group mean mu_a
    b = pm.Normal('beta', mu=mu_b, sd=sigma_b, shape=len(df.county.unique()))

    # Model error
    eps = pm.HalfCauchy('eps', beta=1)

    # Expected value
    # radon_est = a[county_idx] + b[county_idx] * df.floor.values
    radon_est = pm.Deterministic( 'radon_est', a[county_idx] + b[county_idx] * df.floor.values)

    # Data likelihood
    y_like  = pm.Normal('y_like', mu=radon_est, sd=eps, observed=df.log_radon)
    trace   = pm.sample(10000)

pm.summary(trace)
