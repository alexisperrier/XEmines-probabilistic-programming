'''
Iris dataset classification with PyMC3
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm
plt.style.use('seaborn-darkgrid')
# from pymc3 import Model, Normal, HalfNormal


DATA_PATH = '/Users/alexis/amcp/XEmines/2019-PP/data/'

if __name__ == "__main__":

    ''' case 1
    two classes, setosa and versicolor,
    one independent variable or feature, the sepal length
    '''

    iris = sns.load_dataset('iris')
    df      = iris[iris.species.isin(['setosa', 'versicolor'])]

    # [0,0,0,....., 1,1,1,1]
    y       = pd.Categorical(df.species).codes
    cols    = ['sepal_length']
    x       = df[cols].values
    # x = x / np.max(x)

    with pm.Model() as mdl:
        alpha = pm.Normal('alpha', mu = 0, sd = 10)
        beta  = pm.Normal('beta',  mu = 0, sd = 10)
        mu    = alpha + pm.math.dot(x,beta)

        # trace = pm.sample(100, step=pm.Metropolis())

        theta   = pm.Deterministic( 'theta', 1 / ( 1 + pm.math.exp( -mu   )  )   )
        yhat    = pm.Bernoulli( 'yhat' , theta, observed = y)
        bd      = pm.Deterministic('bd' ,-alpha / beta)
        start_  = pm.find_MAP()
        step_   = pm.NUTS()
        trace_  = pm.sample(10000, step_, start_)

    pm.traceplot(trace_[1000:], ['alpha', 'beta','bd'])
    pm.summary(trace_, ['alpha', 'beta','bd'])

    theta   = trace_['theta'].mean(axis=0)
    idx     = np.argsort(x)

    plt.plot(x[idx], theta[idx], color='b', lw=3);
    plt.axvline(trace_0['bd'].mean(), ymax=1, color=''r'')
    bd_hpd = pm.hpd(trace_0['bd'])
    plt.fill_betweenx([0, 1], bd_hpd[0], bd_hpd[1], color='r', alpha=0.5)

    plt.plot(x_0, y_0, 'o', color='k')
    theta_hpd = pm.hpd(trace_0['theta'])[idx]
    plt.fill_between(x_0[idx], theta_hpd[:,0], theta_hpd[:,1], color='b', alpha=0.5)

    plt.xlabel(x_n, fontsize=16)
    plt.ylabel(r'$\theta$', rotation=0, fontsize=16)




# ---
