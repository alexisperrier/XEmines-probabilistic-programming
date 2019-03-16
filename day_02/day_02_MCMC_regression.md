# Jour 2: MCMC et Linear regression

<!-- ------------------------------- -->
## MCMC
<!-- ------------------------------- -->

### Pratique

MCMC Metropolis algorithm
* Markov Chain Monte Carlo for Bayesian Inference - The Metropolis Algorithm

https://www.quantstart.com/articles/Markov-Chain-Monte-Carlo-for-Bayesian-Inference-The-Metropolis-Algorithm

* [Demo] MCMC sampling for dummies
https://twiecki.io/blog/2015/11/10/mcmc-sampling/
https://github.com/twiecki/WhileMyMCMCGentlySamples/blob/master/content/downloads/notebooks/MCMC-sampling-for-dummies.ipynb

+ Approche Markov Chain : theorique

Visual demo of markov chains
http://setosa.io/ev/markov-chains/

https://jeremykun.com/2015/04/06/markov-chain-monte-carlo-without-all-the-bullshit/
Markov Chain Monte Carlo Without all the Bullshit

    * stationary distribution theorem (sometimes called the “Fundamental Theorem of Markov Chains,” and for good reason).
    * for a very long random walk, the probability that you end at some vertex v is independent of where you started!
    * the stationary distribution is a probability distribution \pi such that A \pi = \pi, in other words \pi is an eigenvector of A with eigenvalue 1.
    *  the problem we’re trying to solve is to draw from a distribution over a finite set X with probability function p(x).
    *  The MCMC method is to construct a Markov chain whose stationary distribution is exactly p
    * Now we have to describe the transition probabilities. Let r be the maximum degree of a vertex in this lattice (r=2d). Suppose we’re at vertex i and we want to know where to go next. We do the following:

        * Pick neighbor j with probability 1/r (there is some chance to stay at i).
        * If you picked neighbor j and p(j) \geq p(i) then deterministically go to j.
        * Otherwise, p(j) < p(i), and you go to j with probability p(j) / p(i).
    * We can state the probability weight p_{i,j} on edge (i,j) more compactly as
    \displaystyle p_{i,j} = \frac1r \min(1, p(j) / p(i)) \\ p_{i,i} = 1 - \sum_{(i,j) \in E(G); j \neq i} p_{i,j}


## Monte Carlo
http://barnesanalytics.com/monte-carlo-integration-in-python

Chris Fonnesbeck:
Christopher Fonnesbeck Probabilistic Programming with PyMC3 PyCon 2017
https://youtu.be/5TyvJ6jXHYE?t=1058

## MCMC
Ch3_IntroMCMC_PyMC3.ipynb
MCMC_app.pdf

http://people.duke.edu/~ccc14/sta-663-bootstrap/MCMC.html

Why does Metropolis-Hastings work?¶
1. There is a unique stationary state
2. The stationary state is the posterior probability distribution
http://people.duke.edu/~ccc14/sta-663-bootstrap/MCMC.html#metropolis-hastings-random-walk-algorihtm-for-estimating-the-bias-of-a-coin

# Interpretation
see Probabilistic programming in Python using PyMC3.pdf
posterior analysis


<!-- ------------------------------- -->
##  two types of programming variables: stochastic and deterministic
<!-- ------------------------------- -->
stochastic variables are variables that are not deterministic, i.e., even if you knew all the values of the variables' parameters and components, it would still be random. Included in this category are instances of classes Poisson, DiscreteUniform, and Exponential.

deterministic variables are variables that are not random if the variables' parameters and components were known. This might be confusing at first: a quick mental check is if I knew all of variable foo's component variables, I could determine what foo's value is.


<!-- ------------------------------- -->
# Linear regresssion
<!-- ------------------------------- -->
https://docs.pymc.io/notebooks/getting_started#A-Motivating-Example:-Linear-Regression

1. artificial data generation
2. iris
http://ai.orbifold.net/default/probabilistic-programming-applied-to-the-iris-set-using-pymc3/

3. titanic
4. height vs weight vs gender
http://people.duke.edu/~ccc14/sta-663-bootstrap/PyMC3.html#simple-logistic-model



# GLM
https://docs.pymc.io/notebooks/getting_started#Generalized-Linear-Models


Bayesian Inference with PyMC3 - Part 1
https://blog.applied.ai/bayesian-inference-with-pymc3-part-1/
Create and fit a Bayesian OLS model

# How to debug a model
https://github.com/pymc-devs/pymc3/blob/master/docs/source/notebooks/howto_debugging.ipynb

# Sampler stats
https://pymc3.readthedocs.io/en/stable/notebooks/sampler-stats.html

# Model comparison
https://pymc3.readthedocs.io/en/stable/notebooks/model_comparison.html


## `pymc3.plots`


**PM**

Ch2_MorePyMC_PyMC3.ipynb

## Debugging, visualization

# TD
see (and notebooks)
https://towardsdatascience.com/bayesian-linear-regression-in-python-using-machine-learning-to-predict-student-grades-part-2-b72059a8ac7e




# Further reading
A Beginner's Guide to Markov Chain Monte Carlo, Machine Learning & Markov Blankets
https://skymind.ai/wiki/markov-chain-monte-carlo

A Zero-Math Introduction to Markov Chain Monte Carlo Methods
https://towardsdatascience.com/a-zero-math-introduction-to-markov-chain-monte-carlo-methods-dcba889e0c50

# Good QA on SO
Bayesian logit model - intuitive explanation?
https://stats.stackexchange.com/questions/163034/bayesian-logit-model-intuitive-explanation


Bayes regression: how is it done in comparison to standard regression?
https://stats.stackexchange.com/questions/252577/bayes-regression-how-is-it-done-in-comparison-to-standard-regression


# Alternative formulation using GLM formulas¶
http://people.duke.edu/~ccc14/sta-663-bootstrap/PyMC3.html#alternative-fromulation-using-glm-formulas
