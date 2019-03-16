'''
Binomial
Coin flip with proba p
n trials / tosses
pmf = p(x = k / n, p) with  0 <= k <= n

'''

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
plt.style.use('seaborn-darkgrid')

x = np.arange(0, 40)
ns = [10, 15, 20]
ps = [0.5, 0.5, 0.5]
for n, p in zip(ns, ps):
    pmf = st.binom.pmf(x, n, p)
    plt.plot(x, pmf, '-o', label='n = {}, p = {}'.format(n, p))
plt.xlabel('x', fontsize=14)
plt.ylabel('f(x)', fontsize=14)
plt.legend(loc=1)
plt.show()
