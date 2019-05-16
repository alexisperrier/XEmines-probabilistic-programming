'''
based on
- https://stats.stackexchange.com/questions/353146/forecasting-intermittent-demand-with-pymc3
- https://discourse.pymc.io/t/how-to-set-up-a-custom-likelihood-function-for-two-variables/906/2

'''
import pandas as pd
import numpy as np
import pymc3 as pm
import seaborn as sns
import scipy.stats as stats
from theano import tensor as tt


if __name__ == "__main__":

    ts  = [0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 2, 0, 0, 0, 0,0, 0, 0, 1, 0, 0]
    alpha_ = [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0,0, 0, 0, 1, 0, 0]
    tau_ = [2, 1, 1, 2, 2, 1]


    with pm.Model() as model:
        p   = pm.Beta('p', alpha=1, beta=1)

        mu    = pm.Gamma('mu', alpha=0.001, beta=0.001)
        alpha = pm.Bernoulli('alpha', p, observed = alpha_)
        tau   = pm.Poisson('tau', mu, observed= ts)

        ts_ = pm.Deterministic( 'ts_',  tt.switch(tt.eq(alpha, 0), 0, tau   ) )

        trace = pm.sample(10000)




    def custom_logp(t, d):
        def ll_f(x):
            return - (1/d) * tt.sum(tt.pow(x[t: t + d], 2))
    return ll_f

    def likelihood(value):
        # obs = pm.math.switch(tt.eq(alpha, 1), 0, demand_pos)
        def logp()
        a = value[0]
        b = value[1]

        T.switch( T.eq(a, 0), 0, b   )




model = pm.Model()

with model:
    t = pm.DiscreteUniform('t', lower=0, upper=N-1)
    d = pm.DiscreteUniform('d', lower=1, upper=N-t)

    X_obs = pm.DensityDist('X_obs', custom_logp(t, d), observed=np.zeros(N))




    with pm.Model() as model:
        p           = pm.Beta('p', alpha=1, beta=1)
        alpha       = pm.Bernoulli('alpha', p)

        mu       = pm.Gamma('mu', alpha=0.001, beta=0.001)
        tau      = pm.Poisson('tau', mu)

        # like = pm.DensityDist('like', likelihood(alpha,tau), observed = {'alpha': alpha_,'tau': tau_})
        like = pm.DensityDist('like', likelihood, observed = {'alpha': alpha_,'tau': tau_})

        trace = pm.sample(10000)



    a = pm.Normal('a', mu=0, sd=1, observed=y)
    a = pm.Potential('a', pm.Normal.dist(mu=0, sd=1).logp(y) )







# -------------


import theano

x = np.random.randn(100)
y = x > 0

x_shared = theano.shared(x)
y_shared = theano.shared(y)

with pm.Model() as model:
    coeff = pm.Normal('x', mu=0, sd=1)
    logistic = pm.math.sigmoid(coeff * x_shared)
    pm.Bernoulli('obs', p=logistic, observed=y_shared)
    trace = pm.sample()

x_shared.set_value([-1, 0, 1.])
y_shared.set_value([0, 0, 0]) # dummy values

with model:
    post_pred = pm.sample_posterior_predictive(trace, samples=500)
