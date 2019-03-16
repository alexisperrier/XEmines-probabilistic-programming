# Proba prog Bayes Inference

## Examples

- example of a coin flip: calculate the bias (bayes for hackers p27-28 ? see fig)
- regime change in time series
    - generalize to N regime change detection, how would you do it with ML, Stats ?
lab, teams with different datasets (Stock market, IoT, Social, Demand planning ),  exposé on results?
~ Regime change ?
- A/B testing

We have N visitors, n conversion, observed freq = n/N but not always = to true freqm

# Structure of a PP BI Pymc codebase

1. Specify the priors
* Modeling the variables as stochastic variables, defining the distributions that reflect the assumptions
* parent child:  setting parent variables distributions
* Defining the deterministic interaction between the variables
* add observations (observed = True, value = <samples>)

# process: data generation stories
- What is the best random variable to describe this variable?
- What distribution for the random variable (Poisson => Lambda?, ...)
=> graphical model of observation generation model


2. Model
Group variables into model





* Sampling
* Assessing
* Predicting:

# Pymc variables
- value
- random
@pm.deterministic wrapper


# concepts, examples and illustrations

- why is it called Probabilistic programming: see fig 2 of  Intro to Probabilistic Programming
- simple example of bayes formula: farmer / librarian Bayesian for Hackers p30
- text message data: replace with twitter data ? or personal student data ? p40 BayesHackers
- p44 This type of programming is called probabilistic programming: we create probability models using programming variables as the model’s components.
- Model components are first-class primitives within the PyMC framework.

- visualize prior surface + explanation of how data modifies prior surface => posterior surface.
- exercise p 143, change N size of samples and visualize impact on surface for U and Exp priors. + derive exercise with Normal distributions ?
- Laplace approximation vs variational Bayes vs MCMC ? p 148
- how to implement an asymetric loss function in classic ML, for instance if *estimating a value larger than the true estimate is preferable to estimating a value that is smaller.* p 223
in bayesian estimation we can design our own loss functions.
can we define loss functions in scikit? a priori not possible as it requires forking scikit, writing loss in cython + integrating back

# Regression
- type error in regression
https://discourse.pymc.io/t/type-error-on-regression-problem/2518



# Resources

- [http://dippl.org/](The Design and Implementation of Probabilistic Programming Languages
Noah D. Goodman and Andreas Stuhlmüller )
