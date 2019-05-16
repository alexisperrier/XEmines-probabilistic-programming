import pandas as pd
import numpy as np
import pymc3 as pm
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('sample_sku.csv')
# we have 10 different products
print("SKUs: {}".format(df.sku.unique()))

for sku in df.sku.unique():
    plt.plot( df[df.sku == sku].qty, label = sku )

plt.legend()


sku = df.sku.unique()[2]
plt.plot( list(df[df.sku == sku].qty), label = sku )

ts = list(df[df.sku == sku].qty)
plt.plot(ts)

alpha = 1.0/np.mean(ts)
idx = np.arange(len(ts))

# premier version avec tau ~  DiscreteUniform
with pm.Model() as model:
    lambda_2 = pm.Exponential("lambda_2", alpha)
    lambda_1 = pm.Exponential("lambda_1", alpha)

    tau      = pm.DiscreteUniform("tau", lower=0, upper=50 )

    # model the switch
    lambda_ = pm.Deterministic('lambda_', pm.math.switch(tau > idx, lambda_1, lambda_2))

    # the likelihood
    obs = pm.Poisson("obs", lambda_, observed=ts)

    trace = pm.sample(10000 )
pm.summary(trace, varnames = ['lambda_1', 'lambda_2', 'tau'])
pm.traceplot(trace, varnames = ['tau', 'lambda_1', 'lambda_2'] )

# Deuxieme version avec tau ~ Uniform
with pm.Model() as model:
    lambda_2 = pm.Exponential("lambda_2", alpha)
    lambda_1 = pm.Exponential("lambda_1", alpha)

    tau      = pm.Uniform("tau", lower=10, upper=30 )

    lambda_ = pm.Deterministic('lambda_', pm.math.switch(tau > idx, lambda_1, lambda_2))

    # the likelihood
    obs = pm.Poisson("obs", lambda_, observed=ts)

    trace = pm.sample(10000, step = pm.Metropolis() )

pm.summary(trace, varnames = ['lambda_1', 'lambda_2', 'tau'])
pm.traceplot(trace, varnames = ['tau', 'lambda_1', 'lambda_2'] )
fig, ax = plt.subplots(1,1)
plt.plot(ts)
plt.grid()

# 3 version avec lambda ~N(milieu, 10)

with pm.Model() as model:
    lambda_2 = pm.Normal("lambda_2", mu = 20, sd = 10)
    lambda_1 = pm.Normal("lambda_1", mu = 20, sd = 10)

    tau      = pm.Uniform("tau", lower=0, upper=50 )

    lambda_ = pm.Deterministic('lambda_', pm.math.switch(tau > idx, lambda_1, lambda_2))

    # the likelihood
    obs = pm.Poisson("obs", lambda_, observed=ts)

    trace = pm.sample(10000, step = pm.Metropolis() )

pm.summary(trace, varnames = ['lambda_1', 'lambda_2', 'tau'])
pm.traceplot(trace, varnames = ['tau', 'lambda_1', 'lambda_2'] )
fig, ax = plt.subplots(1,1)
plt.plot(ts)
plt.grid()
