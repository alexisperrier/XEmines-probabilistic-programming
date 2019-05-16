'''
Polling
see https://nbviewer.jupyter.org/github/clausherther/public/blob/master/Dirichlet%20Multinomial%20Example.ipynb#Polling-#1
for the whole notebook
'''

import numpy as np

import scipy.stats as st

import numpy as np
import pymc3 as pm
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":

    '''
    1,447 likely voters were surveyed about their preferences in the upcoming presidential election
    Their responses were:
    Bush: 727
    Dukakis: 583
    Other: 137
    '''

    y = np.asarray([727, 583, 137])
    n = y.sum()
    k = len(y)

    '''
    We, again, set up a simple Dirichlet-Multinomial model and include a Deterministic variable that calculates the value of interest - the difference in probability of respondents for Bush vs. Dukakis.
    '''

    with pm.Model() as polling_model:

        # initializes the Dirichlet distribution with a uniform prior:
        a = np.ones(k)
        theta = pm.Dirichlet("theta", a=a)

        bush_dukakis_diff = pm.Deterministic("bush_dukakis_diff", theta[0] - theta[1])

        likelihood  = pm.Multinomial("likelihood", n=n, p=theta, observed=y)
        polling_trace = pm.sample(1000)



# ------------
