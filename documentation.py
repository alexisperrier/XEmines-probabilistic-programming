# DensityDist

with pm.Model() as model:
    mu = pm.Normal('mu',0,1)
    normal_dist = pm.Normal.dist(mu, 1)
    pm.DensityDist('density_dist',
        normal_dist.logp,
        observed=np.random.randn(100),
        random=normal_dist.random)
    trace = pm.sample(100)

'''
with pm.Model() as model:
    records everything declared in the model object
'''

# Basic
with pm.Model() as model:
    mu  = pm.Normal('mu', mu=0, sd=1)
    obs = pm.Normal('obs', mu=mu, sd=1, observed=np.random.randn(100))

model.basic_RVs
# [mu, obs]
model.free_RVs
# [mu]
model.observed_RVs
# [obs]
model.deterministics
# []
model.logp({'mu': 0})

'''
Every probabilistic program consists of observed and unobserved Random Variables (RVs).
Observed RVs are defined via likelihood distributions,
while unobserved (free) RVs are defined via prior distributions.
Observed RVs are defined just like unobserved RVs but require data to be passed into the observed keyword argument
'''
# Deterministic transforms
with pm.Model():
    x = pm.Normal('x', mu=0, sd=1)
    plus_2 = pm.Deterministic('x plus 2', x + 2)

# In order to sample models more efficiently, PyMC3 automatically transforms bounded RVs to be unbounded.

with pm.Model() as model:
    x = pm.Uniform('x', lower=0, upper=1)
model.free_RVs
# [x_interval__]
# unbounded for more efficient sampling


# --------
