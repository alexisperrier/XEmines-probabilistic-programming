'''
let's simulate some regression dataset and apply PyMC3 to infer the coefficients
'''

import pandas as pd
import numpy as np
import pymc3 as pm

if __name__ == "__main__":

    # -------------------------------------------------------------
    #  step 1: Create dataset using PyMC3
    # y = 1 + 0.5 x
    # -------------------------------------------------------------
    with pm.Model() as generate_model:
        alpha = pm.Uniform('alpha', lower = 0, upper = 100)
        x = pm.Normal('x', mu = 10, sd = 2)
        y = pm.Deterministic('y', alpha + 5 * x)
        gtrace = pm.sample(5000, step=pm.NUTS())

    X = pd.DataFrame( columns = ['x','y'] )
    X['x'] = gtrace['x']
    X['y'] = gtrace['y']

    n_samples = X.shape[0]

    fig, ax = plt.subplots(1,1, figsize = (12,6))
    plt.hist(gtrace['x'], bins = 100, label = 'x', alpha= 0.8)
    plt.hist(gtrace['alpha'], bins = 100, label = 'alpha', alpha= 0.5)
    plt.hist(gtrace['y'], bins = 100, label = 'y', alpha= 1)
    plt.legend()
    plt.show()

    # step 2: inference
    # assume y ~ N(mu X , sigma^2)
    XX = X.sample(n = 1000)
    with pm.Model() as inference_model:
        # Define priors
        intercept   = pm.Normal('intercept', 0, sd=20)
        sigma       = pm.HalfCauchy('sigma', beta=10, testval=1.)
        x_coeff     = pm.Normal('x_coeff', 0, sd=20)

        # Define likelihood
        lklh = pm.Normal('lklh', mu=intercept + x_coeff * XX.x, sd=sigma, observed=XX.y)

        # Inference!
        trace = pm.sample(5000, step=pm.NUTS())

    Z = pd.DataFrame( columns = ['intercept','sigma', 'x_coeff'] )
    Z['intercept'] = trace['intercept']
    Z['sigma'] = trace['sigma']
    Z['x_coeff'] = trace['x_coeff']

pm.traceplot(trace)
fig, ax = plt.subplots(1,1, figsize = (12,6))
plt.hist(likelihood.distribution.random(), bins = 100, label = 'likelihood')
plt.legend()
plt.show()

plt.hist(lklh.distribution.random(), bins = 100, label = 'likelihood')

res = []
for i in range(4):
    data = list(lklh.distribution.random())
    plt.hist(data, bins = 30)
    res += data

plt.show()

fig, ax = plt.subplots(1,1, figsize = (12,6))
plt.hist(gtrace['x'], bins = 100, label = 'x', alpha= 0.8)
plt.hist(gtrace['alpha'], bins = 100, label = 'alpha', alpha= 0.5)
plt.hist(gtrace['y'], bins = 100, label = 'y', alpha= 1)
plt.hist(res, bins = 100, label = 'likelihood')
plt.legend()
plt.show()


plt.show()

    with pm.Model() as generate_model:
        coeff_mu = pm.Uniform('coeff_x', lower = 0, upper = 20)
        beta    = pm.Normal('beta', mu = 0, sd = 1)
        x = pm.Normal('x', mu = np.mean(X.x), sd = np.std(X.x))
        relation = pm.Deterministic('y', beta + coeffs[0] * x)

        intercept = pm.Normal('Intercept', 0, sd=20)
    x_coeff = pm.Normal('x', 0, sd=20)
    price_coef = pm.Normal('price', 0, sd=20)

    # Define likelihood
    likelihood = pm.Normal('y',
                              pm.math.sigmoid(intercept+x_coeff*df['BEDRMS']+price_coef*df['COSTMED']),
                              observed=df['OWN'])
    WTP=pm.Deterministic('WTP',-x_coeff/price_coef)

        y = pm.Normal()

        trace = pm.sample(100, step=pm.Metropolis())






# -------------------------
