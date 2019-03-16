'''
Nicole Carlson | A Quickstart Guide to PyMC3
https://www.youtube.com/watch?v=rZvro4-nFIk
'''

import pandas as pd
import theano.tensor as T
import pymc3 as pm
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    num_features = 2
    num_observed = 1000
    num_steps    = 10000

    # generate data
    alpha_a = np.random.normal(size = 1)
    betas_a = np.random.normal([-1,1],size = num_features)
    X_train = np.random.normal(size = (num_features, num_observed))
    y_a = alpha_a + np.dot(X_train.T, betas_a) + np.random.normal(1, 2,  size = num_observed)

    # model

    with pm.Model() as lin_reg_model2:
        alpha = pm.Normal('alpha', mu = 0, tau = 0.01, shape = 1)
        betas = pm.Normal('betas', mu = 0, tau = 0.01, shape = (1, num_features) )

        s = pm.HalfNormal('s', tau = 1)
        temp = alpha + T.dot(betas, X_train)
        y = pm.Normal('y', mu = temp, tau = 0.01, observed = y_a)

    with lin_reg_model2:

        step = pm.NUTS()
        nuts_trace = pm.sample(num_steps, step)

    pm.traceplot(nuts_trace[1000:])
    print(alpha_a)
    print(betas_a)
    print(pm.summary(nuts_trace[1000:]))

# pass numpy arrays (no dataframes) with all numerical values (no booleans for instance)

# Other syntax
# with pm.Model() as model:

# how to improve model ?
#

# ----------------
