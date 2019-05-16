import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import seaborn as sns

# Cauchy
x = np.linspace(0, 5, 200)
for b in [0.5, 1.0, 2.0]:
    pdf = st.cauchy.pdf(x, scale=b)
    plt.plot(x, pdf, label=r'$\beta$ = {}'.format(b))
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.legend(loc=1)
plt.show()

# load df
    # df = pd.read_csv('../../data/radon.csv')

county_names = df.county.unique()
county_idx = df['county_code'].values

with pm.Model() as hierarchical_model:
    # Hyperpriors
    mu_a    = pm.Normal('mu_alpha', mu=0., sd=1)
    sigma_a = pm.HalfCauchy('sigma_alpha', beta=1)
    mu_b    = pm.Normal('mu_beta', mu=0., sd=1)
    sigma_b = pm.HalfCauchy('sigma_beta', beta=1)

    # Intercept for each county, distributed around group mean mu_a
    a = pm.Normal('alpha', mu=mu_a, sd=sigma_a, shape=len(df.county.unique()))
    # Intercept for each county, distributed around group mean mu_a
    b = pm.Normal('beta', mu=mu_b, sd=sigma_b, shape=len(df.county.unique()))

    # Model error
    eps = pm.HalfCauchy('eps', beta=1)

    # Expected value
    # radon_est = a[county_idx] + b[county_idx] * df.floor.values
    radon_est = pm.Deterministic( 'radon_est', a[county_idx] + b[county_idx] * df.floor.values)

    # Data likelihood
    y_like  = pm.Normal('y_like', mu=radon_est, sd=eps, observed=df.log_radon)
    trace   = pm.sample(10000)

pm.summary(trace)
trace.varnames

pm.traceplot(trace, varnames= ['mu_alpha'])

pm.summary(trace, varnames = ['alpha'])
pm.traceplot(trace, varnames = ['alpha'])

pm.traceplot(trace, varnames = ['beta'])
pm.summary(trace, varnames = ['beta'])
res = pm.summary(trace, varnames = ['beta'])
res.head()
plt.hist(res['mean'])
res = pm.summary(trace, varnames = ['alpha'])
plt.hist(res['mean'])
plt.show()
