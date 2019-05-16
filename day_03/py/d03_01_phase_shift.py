'''
Inspired from https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter1_Introduction/Ch1_Introduction_PyMC3.ipynb
'''

import pandas as pd
import numpy as np
import pymc3 as pm
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":

    df = pd.read_csv('txt_data.csv')
    df.head()


    # plot
    fig, ax = plt.subplots(1,1, figsize = (12,6))
    # plt.bar(df.index, df.ntxt)
    sns.barplot(x = df.index, y = df.ntxt)
    plt.title('number of text sent per day')
    plt.ylabel("count of text-msgs received")
    plt.tight_layout()


    '''
    Do you notice a change?
    '''


    alpha = 1.0/df.ntxt.mean()
    idx = np.arange(df.shape[0]) # Index [0, 73]

    with pm.Model() as model:
        lambda_1 = pm.Normal("lambda_1", mu = 0, sd = 100)
        lambda_2 = pm.Normal("lambda_2", mu = 0, sd = 100)
        # lambda_2 = pm.Exponential("lambda_2", alpha)

        tau      = pm.DiscreteUniform("tau", lower=0, upper=100)

        # model the switch
        # if tau > i then lambda_2 else lambda_1

        lambda_ = pm.Deterministic('lambda_', pm.math.switch(tau > idx, lambda_1, lambda_2))

        # the likelihood
        obs = pm.Poisson("obs", lambda_, observed=df.ntxt)

        trace = pm.sample(10000, step = pm.Metropolis() )


    sns.distplot(trace['tau'])
    # only three or four days make any sense as potential transition points

    # --------------------------------------------------------
    #  now for some real data
    # --------------------------------------------------------

    df = pd.read_csv('sample_sku.csv')
    # we have 10 different products
    print("SKUs: {}".format(df.sku.unique()))

    fig,ax = plt.subplots(1,1, figsize = (12,6))
    sns.lineplot(x = 'month', y = 'qty', hue = 'sku', data = df  )

    fig,ax = plt.subplots(1,1, figsize = (12,6))
    for sku in df.sku.unique():
        plt.plot( list(df[df.sku == sku].qty), label = sku )

    plt.legend()

    # sku = df.sku.unique()[9]

    alpha = 1.0/np.mean(ts)
    idx = np.arange(len(ts))

    with pm.Model() as model:
        lambda_2 = pm.Exponential("lambda_2", alpha)
        lambda_1 = pm.Exponential("lambda_1", alpha)

        tau      = pm.DiscreteUniform("tau", lower=0, upper=50 )

        # model the switch
        # if tau > i then lambda_2 else lambda_1

        lambda_ = pm.Deterministic('lambda_', pm.math.switch(tau > idx, lambda_1, lambda_2))

        # the likelihood
        obs = pm.Poisson("obs", lambda_, observed=ts)

        trace = pm.sample(10000 )




# ------------------------
