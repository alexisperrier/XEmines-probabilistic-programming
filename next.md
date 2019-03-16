# Next: regression lineaire
- comment verifier la qualité du modele ?


- suivre l'exemple de regression lineaire avec GLM:
https://www.quantstart.com/articles/Bayesian-Linear-Regression-Models-with-PyMC3

- generate data to do posterior predictive checks
https://pymc3.readthedocs.io/en/stable/notebooks/posterior_predictive.html

- suivre
https://alexioannides.com/2018/11/07/bayesian-regression-in-pymc3-using-mcmc-variational-inference/
Bayesian Regression in PYMC3 using MCMC & Variational Inference


study other diagnostic plots see: https://eigenfoo.xyz/bayesian-modelling-cookbook/
pm.densityplot(trace)
pm.energyplot(trace)
pm.autocorrplot(trace)
pm.forestplot(trace)
pm.plot_posterior(trace)


* pm.summary(trace)
Rhat => 1
Gelman–Rubin statistic?

* pm.sample parameters
try to pm.sample with draws=1, tune=500, and discard_tuned_samples=False

* WARNING:pymc3:The acceptance probability does not match the target. It is 0.9372216531503921, but should be close to 0.8. Try to increase the number of tuning steps.

* regression on media data

* logistic on ?

* gaussian on non linearly separable plots?
