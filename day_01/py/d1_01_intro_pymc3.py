'''
'''
import numpy as np
import scipy.stats as st

import pymc3 as pm

import seaborn as sns
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------
#  Random variables, prior
# ------------------------------------------------------------------------------

'''
Define a random variable following  N(1,1) distribution
'''

with pm.Model() as model:
    x  = pm.Normal('x', mu=1, sd=1)

'''
Get N=1000 random samples of that variable
'''
data = x.random(size = 1000)
sns.distplot( data  )

'''
Random Variables in the model
'''
model.vars

'''
log probability density function
'''
model.logp({'x': 1})


'''
Let's now observe some data
    data = np.random.randn(100)
Assume that
- the data follows a Normal distribution with sd = 1 and unknown mean mu
- the mean mu also follow a normal distribution (0,1)

'''

data = np.random.randn(100)

with pm.Model() as model:

    obs = pm.Normal('obs', mu=0, sd=1, observed=data)

'''
Note: The obs var appears as model.observed_RVs, but no longer has a random method

The x variable is the prior, the obs variable is the likelihood = p( data / x  )
We want the distribution of the posterior:
p(x / data) = p( data / x  ) * p(x) / p(data)
for that we sample the model

try with U(0,10) then U(-1,1)
'''

with pm.Model() as model:
    # prior
    x  = pm.Uniform('x', lower=-10, upper=10)

    obs = pm.Normal('obs', mu=x, sd=1, observed=data)

    trace = pm.sample()

'''
pm.summary(trace)
We see that the mean of x, knowing the data, is != 0.
=> n_eff, RHat, confidence interval
'''

'''
What happens if we input more samples in the dataset
'''
data = np.random.randn(1000)
with pm.Model() as model:
    x  = pm.Uniform('x', lower=-10, upper=10)
    obs = pm.Normal('obs', mu=x, sd=1, observed=data)
    trace = pm.sample(10000)

'''
same results
'''

'''
how many samples of the posterior distribution ? 1000 *4
let's sample more: 10000
much better looking traceplot
'''

data = np.random.randn(100)
with pm.Model() as model:
    mu  = pm.Normal('mu', mu=0, sd=1)
    obs = pm.Normal('obs', mu=mu, sd=1, observed=data)
    trace = pm.sample(10000)


'''
Now generate some data, normally distributed, sd, mean whatever you want
with
    import scipy.stats as st
    data = st.norm.rvs(loc = <mean>, scale = <standard deviation>, size = 1000)

And find the distribution mean and sd with pymc assuming both mu and sd follow a Uniform distribution
'''



# Generate 1000 samples of a  Normal(2,10) distrobution
data = st.norm.rvs(loc = 2, scale = 10, size = 1000)

with pm.Model() as model:
    mu  = pm.Uniform('mu', lower = 0, upper = 20)
    sd  = pm.Uniform('sd', lower = 0, upper = 100)
    obs = pm.Normal('obs', mu=mu, sd=sd, observed=data)
    trace = pm.sample(10000)


'''
Change the prior distributions to Normal
'''

data = st.norm.rvs(loc = 2, scale = 1, size = 1000)
with pm.Model() as model:
    mu  = pm.Normal('mu', mu = 0, sd = 1)
    sd  = pm.HalfNormal('sd', mu = 0, sd = 1)
    obs = pm.Normal('obs', mu=mu, sd=sd, observed=data)
    trace = pm.sample(10000, step = pm.Metropolis())


'''
Chain N failed???
'''

'''
Instead use HalfNormal for sd
see also http://people.duke.edu/~ccc14/sta-663-bootstrap/PyMC3.html#estimating-mean-and-standard-deviation-of-normal-distribution
'''

data = st.norm.rvs(loc = 2, scale = 10, size = 1000)
with pm.Model() as model:
    mu  = pm.Normal('mu', mu = 0, sd = 1)
    sd  = pm.HalfNormal('sd', sd = 1)
    obs = pm.Normal('obs', mu=mu, sd=sd, observed=data)
    trace = pm.sample(10000)





# ----------------
