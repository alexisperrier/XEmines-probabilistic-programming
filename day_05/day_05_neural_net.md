# Jour 5: Gaussian processes


* Fitting Gaussian Process Models in Python
by Chris Fonnesbeck on March 8, 2017
https://blog.dominodatalab.com/fitting-gaussian-process-models-python/

comparison of gaussian process in scikit and pymc3


# Exemple: 2 gaussiennes
http://people.duke.edu/~ccc14/sta-663-bootstrap/PyMC3.html#estimating-mean-and-standard-deviation-of-normal-distribution

2 Gaussiennes
    - differentes tailles d'echantillons
    - comment estimer mu et sigma


# Demo des 2 gaussiennes avec PyMC3

case: 2 gaussians, not same number of samples, estimate mu and sigma with pymc3
[notebook](https://docs.pymc.io/notebooks/marginalized_gaussian_mixture_model.html)

[Ch3_IntroMCMC_PyMC3.ipynb](Ch3_IntroMCMC_PyMC3.ipynb)


**[TD]** Gaussian Mixture
https://docs.pymc.io/notebooks/gaussian_mixture_model.html

How to build a Gaussian Mixture Model
https://hub.packtpub.com/how-to-build-a-gaussian-mixture-model/

Avec extension de binomial a categorical + extension de beta = dirichlet

=> Comment valider les resultats ?
=> visualisation




# Further reading
Fitting Gaussian Process Models in Python
https://blog.dominodatalab.com/fitting-gaussian-process-models-python/

A Gaussian process generalizes the multivariate normal to infinite dimension. It is defined as an infinite collection of random variables, with any marginal subset having a Gaussian distribution.

Gaussian Mixture Model
https://brilliant.org/wiki/gaussian-mixture-model/
math behind Gaussian mixture models

##




Variational Inference ADVI


**AM**

## Resources, ecosystem and who to read
Inteview with Thomas Wiecki about PyMC and probabilistic programming


- prediction generation
- missing values
**PM**
- retail credit case

- supply chain ? https://peadarcoyle.com/2018/12/18/applications-of-bayesian-statistics-supply-chain/


# Future

https://twiecki.github.io/blog/2018/08/13/hierarchical_bayesian_neural_network/
Bayesian Neural Network in PyMC3
https://gist.github.com/anonymous/9287a213fe188a79d7d7774eef79ad4d

# Theano

see http://localhost:8888/notebooks/day_02/Ch2_MorePyMC_PyMC3.ipynb#Theano
