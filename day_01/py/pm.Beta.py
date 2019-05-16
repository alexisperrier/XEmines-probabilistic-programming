'''
Beta
Beta distribution is a conjugate prior for the parameter 𝑝 of the binomial distribution.

𝑓(𝑥∣𝛼,𝛽)=𝑥^𝛼−1(1−𝑥)^𝛽−1 / 𝐵(𝛼,𝛽)

'''

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
plt.style.use('seaborn-darkgrid')
alphas = [.5, 5., 1., 2., 2.]
betas = [.5, 1., 3., 2., 5.]


x = np.linspace(0, 1, 200)
fig,ax = plt.subplots(1,3, figsize = (15,5))
plt.subplot(1,3,3)
alphas = [5, 5, 5, 5, 5]
betas = [.5, 1., 2., 3.,  5.]

for a, b in zip(alphas, betas):
    pdf = st.beta.pdf(x, a, b)
    plt.plot(x, pdf, label=r'$\alpha$ = {}, $\beta$ = {}'.format(a, b))
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.ylim(0, 4.5)
plt.legend(loc=9)

plt.subplot(1,3,1)

alphas = [1,1,1,1,1]
betas = [.5, 1., 2., 3.,  5.]

for a, b in zip(alphas, betas):
    pdf = st.beta.pdf(x, a, b)
    plt.plot(x, pdf, label=r'$\alpha$ = {}, $\beta$ = {}'.format(a, b))
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.ylim(0, 4.5)
plt.legend(loc=9)

plt.subplot(1,3,2)
alphas = [2,2,2,2,2]
betas = [.5, 1., 2., 3.,  5.]

for a, b in zip(alphas, betas):
    pdf = st.beta.pdf(x, a, b)
    plt.plot(x, pdf, label=r'$\alpha$ = {}, $\beta$ = {}'.format(a, b))
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.ylim(0, 4.5)
plt.legend(loc=9)

plt.show()
