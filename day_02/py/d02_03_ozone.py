'''
Ozone regression
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm
plt.style.use('seaborn-darkgrid')


DATA_PATH = '/Users/alexis/amcp/XEmines/2019-PP/data/'

if __name__ == "__main__":
    df = pd.read_csv('ozone.csv')
    df = df.dropna()[['Ozone', 'Solar.R', 'Wind', 'Temp']].rename(columns = {'Solar.R': 'Solar'})
    df.head()

    fig, ax = plt.subplots(2,2)
    i = 0
    for c in df.columns:
        i += 1
        plt.subplot(2,2,i)
        sns.distplot(df[c], 30)

    plt.tight_layout()
    plt.show()

    # ca resssemble pas a du gaussien , mais commencons par un model simple tout gaussien


    df.corr()
    # montre correlation avec Ozone plus importante
    sns.jointplot( df.Temp, df.Ozone  )

    #  modelisation
    # df.Temp = alpha + beta *  df.Ozone + sigma

    with pm.Model() as model:
        # coefficient de la regression
        alpha   = pm.Normal('alpha', mu=0, sd=10)
        beta    = pm.Normal('beta', mu=0, sd=10)
        # bruit
        sigma   = pm.HalfNormal('sigma', sd=1)

        # Expected value of outcome
        # deterministic random variable
        mu = alpha + beta * df.Ozone
        # observed stochastic = likelihood
        # This creates parent-child relationships between the likelihood and these two variables
        Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=df.Temp)


    with model:
        # obtain starting values via MAP
        step = pm.NUTS()
        # step = pm.Metropolis()
        # draw 2000 posterior samples
        # start = pm.find_MAP()
        trace = pm.sample(10000, step )

    pm.traceplot(trace)
    pm.summary(trace)
    # model.plot_posterior_predictive(trace)



    # -------------------------------------------------------------
    #  Posterior Predictive Checks
    # -------------------------------------------------------------
    map_estimate = pm.find_MAP(model=model)
    print(map_estimate)

    # We can give the start coefficients to the sampling algo
    with model:
        start = find_MAP()
        step = pm.NUTS()
        trace = pm.sample(10000, step = step, start = start )


    # change the prior
    # should show that:
    # "the posterior distribution is essentially invariant under reasonable changes in the prior"


    # -------------------------------------------------------------
    #  Posterior Predictive Checks
    # -------------------------------------------------------------
    '''
    Creates 1000 datasets based on the model we trained
    '''
    ppc = pm.sample_ppc(trace, model=model, samples = 1000)
    # Compare the distribution of the original dataset and the posterior distribution
    fig, ax = plt.subplots(1,1)
    plt.hist(df.Temp, bins = 20)
    plt.hist([n.mean() for n in ppc['Y_obs']], bins = 100, alpha = 0.4)
    plt.show()

    '''
    We see that our model is not really fitting the original dataset
    '''

    # -------------------------------------------------------------
    #  Stats model
    # -------------------------------------------------------------
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    import statsmodels.api as sm
    X = df.Ozone.copy()
    X = sm.add_constant(X)
    ols = sm.OLS(df.Temp, X).fit()

    ols.summary()
    # similar coefficients


    # ------------------------------------------------------------
    #  now with 2 predictors
    # ------------------------------------------------------------
    with pm.Model() as model_n:
        # coefficient de la regression
        alpha   = pm.Normal('alpha', mu=0, sd=10)
        beta    = pm.Normal('beta', mu=0, sd=10, shape = 2)
        sigma   = pm.HalfNormal('sigma', sd=1)

        # likelihood
        mu = pm.Deterministic('mu', alpha + beta[0] * df.Ozone + beta[1] * df.Wind)
        # pm.Deterministic(
        Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=df.Temp)
        trace_n = pm.sample(50000, pm.NUTS())

    pm.traceplot(trace_n)
    pm.summary(trace_n)

    '''
    beta__1 HDI : -0.337509   0.449592
    can beta__1 = 0 ?
    how do we accept beta__1 as 0 ? => HDI
    '''

    '''
    In Kruschke: R^2 is calculated, see paper
    '''


    # -----------------------
    X = df[['Ozone','Wind']]
    X = sm.add_constant(X)
    ols = sm.OLS(df.Temp, X).fit()

    ols.summary()

    '''
    we see that the Wind coefficient is not relevant (statsmodel)
    how do we diagnose that with pymc?
    '''

    # ------------------------------------------------------------
    #  Normalize the data
    # Better results =>  beta_1 is not longer squished below beta_0
    # But still around 0
    #  Normalize data and observe beta_1 - beta_2 (see Kruschke)
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    for c in df.columns:
        df[c] = df[c] / np.max(df[c])

    '''
    comparé a OLS, on retrouve les memes intervales de confiance
    '''

    # ------------------------------------------------------------
    #  now with t student for Temp
    # https://docs.pymc.io/notebooks/GLM-robust-with-outlier-detection.html#Create-Robust-Model:-Student-T-Method
    # ------------------------------------------------------------

    for c in df.columns:
        df[c] = df[c] / np.max(df[c])

    with pm.Model() as model_t:
        # coefficient de la regression
        alpha   = pm.Normal('alpha', mu=0, sd=10)
        beta    = pm.Normal('beta', mu=0, sd=10, shape = 2)
        sigma   = pm.HalfNormal('sigma', sd=10)
        nu   = pm.HalfNormal('nu', sd=10)
        mu = alpha + beta[0] * df.Ozone + beta[1] * df.Wind
        Y_obs = pm.StudentT('Y_obs', nu=nu, mu=mu, sd=sigma, observed=df.Temp)
        Y_pred = pm.StudentT('Y_pred', nu=nu, mu=mu, sd=sigma, shape=df.shape[0])
        # Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=df.Temp)
        trace_t = pm.sample(10000, pm.NUTS())

    pm.traceplot(trace_t)
    pm.summary(trace_t)
    '''
    on retrouve les meme alpha, beta,
    mais on a nu en plus, commet interpreter?

    '''

    '''
    Quand on rajoute:
    Y_pred = trace_t.get_values('Y_pred').ravel()
    on peut voir la tete de la distroibution posterieure
    see this thread https://stats.stackexchange.com/questions/169223/how-to-generate-the-posterior-predictive-distribution-for-hierarchal-model-in-py
    '''

    Y_pred = trace_t.get_values('Y_pred').ravel()
    plt.hist(Y_pred, bins = 100)

    # et comparer a la distribution initiale
    plt.hist(df.Temp, bins = 20)


    '''
    compare forest plot for N et T
    '''
    pm.forestplot(trace_t, varnames = ['alpha', 'beta', 'sigma'])
    pm.forestplot(trace_n, varnames = ['alpha', 'beta', 'sigma'])

    # ------------------------------------------------------------
    #  Posterior Predictive Checks
    # pas concluant
    # ------------------------------------------------------------

    #  3 variables
    with pm.Model() as model_n:
        # coefficient de la regression
        alpha   = pm.Normal('alpha', mu=0, sd=10)
        beta    = pm.Normal('beta', mu=0, sd=10, shape = 3)
        sigma   = pm.HalfNormal('sigma', sd=1)

        # likelihood
        mu = pm.Deterministic('mu', alpha + beta[0] * df.Ozone + beta[1] * df.Wind + beta[2] * df.Solar)
        # pm.Deterministic(
        Y_pred = pm.Normal('Y_pred', mu=mu, sd=sigma, shape = df.shape[0])
        Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=df.Temp)
        trace_n = pm.sample(15000, pm.NUTS())

    pm.traceplot(trace_n, varnames = ['alpha', 'beta', 'sigma'])
    pm.summary(trace_n, varnames = ['alpha', 'beta', 'sigma'])


    # -------------------------------------------------------------------------
    #  from pymc3.glm import GLM
    # -------------------------------------------------------------------------
    from pymc3.glm import GLM
    with pm.Model() as model_glm:
        GLM.from_formula('Temp ~ Solar + Wind + Ozone', df)
        trace_g = pm.sample()

    # compare 2 models
    pm.stats.loo(trace_t, model_t)
    pm.stats.loo(trace_n, model_n)
    pm.stats.waic(trace_t, model_t)

    pm.stats.compare( {model_n: trace_n, model_t: trace_t}    )

    # ???
    pm.stats.r2_score(trace_t, model_t)

    '''
    http://stronginference.com/bayes-factors-pymc.html
    https://stats.stackexchange.com/questions/20729/best-approach-for-model-selection-bayesian-or-cross-validation
    https://jakevdp.github.io/blog/2015/08/07/frequentism-and-bayesianism-5-model-selection/
    https://stats.stackexchange.com/questions/161082/bayesian-model-selection-in-pymc3/166383
    '''


    # ------------------------------------------------------------
    #  Predictions
    # ------------------------------------------------------------
    new_observation = df.loc[0][['Ozone','Wind','Solar']]
    mu = alpha + beta_s dot observation + epsilon
    sd = sigma

    => plt normal(mu, sigma)
    compare mean of new distribution with real score !



        # Dictionary of all sampled values for each parameter
        var_dict = {}
        for variable in trace.varnames:
            var_dict[variable] = trace[variable]

        # Standard deviation
        sd_value = var_dict['sd'].mean()

        # Results into a dataframe
        var_weights = pd.DataFrame(var_dict)

        # Align weights and new observation
        var_weights = var_weights[new_observation.index]

        # Means of variables
        var_means = var_weights.mean(axis=0)

        # Mean for observation
        mean_loc = np.dot(var_means, new_observation)

        # Distribution of estimates
        estimates = np.random.normal(loc = mean_loc, scale = sd_value,
                                     size = 1000)

        # Plot the estimate distribution
        plt.figure(figsize(8, 8))
        sns.distplot(estimates, hist = True, kde = True, bins = 19,
                     hist_kws = {'edgecolor': 'k', 'color': 'darkblue'},
                    kde_kws = {'linewidth' : 4},
                    label = 'Estimated Dist.')
        # Plot the mean estimate
        plt.vlines(x = mean_loc, ymin = 0, ymax = 5,
                   linestyles = '-', colors = 'orange', linewidth = 2.5)
        plt.title('Density Plot for New Observation');
        plt.xlabel('Grade'); plt.ylabel('Density');

        # Estimate information
        print('Average Estimate = %0.4f' % mean_loc)
        print('5%% Estimate = %0.4f    95%% Estimate = %0.4f' % (np.percentile(estimates, 5),
                                           np.percentile(estimates, 95)))


 # -------------------
