import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import pymc3 as pm
import seaborn as sns


if __name__ == "__main__":

    # -------   --------------------
    #  Poisson distribution
    # ---------------------------

    '''
    1. Examples of Poisson for different value of Lambda
    '''
    x = np.arange(0, 15)

    fig,ax = plt.subplots(1,1)
    for m in [0.5, 3, 8]:
        pmf = st.poisson.pmf(x, m)
        plt.plot(x, pmf, '-o', label='$\lambda$ = {}'.format(m))
    plt.title('Poisson distribution for different  $\lambda$')
    plt.show()



    '''
    - Generate a dataset that follows a Poisson distribution
    - Boostrap the data
    - Use PyMC3 to sample the data
    - Compare
    '''

    lambda_ = 5
    data = st.poisson.rvs(lambda_, size =200)
    plt.hist(data)
    # bootstrap : sample with replacement
    sample_count = 1000
    bstrap = np.random.choice(data, sample_count, replace = True)

    plt.subplots(1,1)
    sns.distplot(bstrap)
    plt.vlines(5, 0, 1,  label = 'Real value {}'.format(lambda_)  )
    plt.vlines(np.mean(bstrap), 0, 1,  linestyle =':', label = 'mean of bootstraped data {:.3f}'.format(np.mean(bstrap))  )
    sns.distplot(data)
    plt.vlines(np.mean(data), 0, 1, linestyle = ':', color = 'red', label = 'mean of original data {:.3f}'.format(np.mean(data))  )
    plt.legend()




    # Now with PyMC3

    with pm.Model() as model:

        lambda_ = pm.HalfNormal('lambda_',  sd = 10)
        # lambda_ = pm.Uniform('lambda_', lower = 0, upper = 1 )
        obs = pm.Poisson('obs',  lambda_, observed = data)
        trace = pm.sample(10000)

    pm.summary(trace)

    plt.subplots(1,1, figsize=(12,6))
    sns.distplot(x, label = 'boostrapped')
    plt.vlines(5, 0, 1,  label = 'Real value {}'.format(lambda_)  )
    plt.vlines(np.mean(x), 0, 1,  linestyle =':', label = 'mean of bootstraped data {:.3f}'.format(np.mean(x))  )
    sns.distplot(data,  label = 'original data')
    plt.vlines(np.mean(data), 0, 1, linestyle = ':', color = 'red', label = 'mean of original data {:.3f}'.format(np.mean(data))  )
    sns.distplot(trace['lambda_'], label = 'Bayes estimate')
    plt.vlines(np.mean(trace['lambda_']), 0, 1,  linestyle = ':',color = 'blue', label = 'mean of lambda {:.3f}'.format(np.mean(trace['lambda_']))  )
    plt.legend()

    # sensiblement meilleur pour l'estimateur bayesien

    '''
    Generate 1000 samples
    Increase subset size: 10, 50, 100, 500, 1000
    Compare accuracy of different estimator
    Impact of sample size?
    '''

Scientists rise up against statistical significance
Valentin Amrhein, Sander Greenland, Blake McShane and more than 800 signatories call for an end to hyped claims and the dismissal of possibly crucial effects.







# ---------------
