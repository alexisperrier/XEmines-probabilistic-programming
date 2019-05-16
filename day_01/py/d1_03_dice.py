'''
# Dice, Polls & Dirichlet Multinomials
https://calogica.com/python/pymc3/2018/11/27/dice-polls-dirichlet-multinomial.html

'''
import numpy as np

import scipy.stats as st

import numpy as np
import pymc3 as pm
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":


    y = np.asarray([20,  21, 17, 19, 17, 28])
    k = len(y)
    p = 1/k
    n = y.sum()


    sns.barplot(x=np.arange(1, k+1), y=y);


    with pm.Model() as dice_model:

        # initializes the Dirichlet distribution with a uniform prior:
        a = np.ones(k)

        theta = pm.Dirichlet("theta", a=a)

        # Since theta[5] will hold the posterior probability of rolling a 6
        # we'll compare this to the reference value p = 1/6
        # six_bias = pm.Deterministic("six_bias", theta[k-1] - p)

        results = pm.Multinomial("results", n=n, p=theta, observed=y)

    # Let’s draw 1,000 samples from the joint posterior using the default NUTS sampler:

    with dice_model:
        dice_trace = pm.sample(draws=1000)

    pm.traceplot(dice_trace, combined=True, lines={"theta": p})

    axes = pm.plot_posterior(dice_trace, varnames=["theta"], ref_val=np.round(p, 3))
    for i, ax in enumerate(axes):
        ax.set_title(f"{i+1}")


    # plot the probability of our dice being biased towards 6, by comparing theta[Six] to p



# ---------------
