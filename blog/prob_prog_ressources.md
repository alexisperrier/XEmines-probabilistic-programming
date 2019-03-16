resources

# Sampling

## MCMC

* Markov Chain Monte Carlo for Bayesian Inference - The Metropolis Algorithm

https://www.quantstart.com/articles/Markov-Chain-Monte-Carlo-for-Bayesian-Inference-The-Metropolis-Algorithm

In this article we introduce the main family of algorithms, known collectively as Markov Chain Monte Carlo (MCMC), that allow us to approximate the posterior distribution as calculated by Bayes' Theorem. In particular, we consider the Metropolis Algorithm, which is easily stated and relatively straightforward to understand.

MCMC Metropolis 1953

Begin the algorithm at the current position in parameter space (θcurrent)

* Propose a "jump" to a new position in parameter space (θnew)
* Accept or reject the jump probabilistically using the prior information and available data
* If the jump is accepted, move to the new position and return to step 1
* If the jump is rejected, stay where you are and return to step 1
* After a set number of jumps have occurred, return all of the accepted positions

The main difference between MCMC algorithms occurs in how you jump as well as how you decide whether to jump.

The Metropolis algorithm uses a normal distribution to propose a jump.
A normal distribution is a good choice for such a proposal distribution (for continuous parameters) as, by definition, it is more likely to select points nearer to the current position than further away. However, it will occassionally choose points further away, allowing the space to be explored.

jump decision: p=P(θnew)/P(θcurrent)

We then generate a uniform random number on the interval [0,1]. If this number is contained within the interval [0,p] then we accept the move, otherwise we reject it.

P(θnew|D)P(θcurrent|D)=P(D|θnew)P(θnew)P(D)P(D|θcurrent)P(θcurrent)P(D)=P(D|θnew)P(θnew)P(D|θcurrent)P(θcurrent)


* MCMC sampling for dummies
https://twiecki.io/blog/2015/11/10/mcmc-sampling/

MCMC generates samples from the posterior distribution by constructing a reversible Markov-chain that has as its equilibrium distribution the target posterior distribution.

This blog post is an attempt at trying to explain the intuition behind MCMC sampling (specifically, the random-walk Metropolis algorithm).

constructing a Markov chain to do Monte Carlo approximation

twiecki shows how to program MCMC in the case of Gaussian prior / conjugate posterior
A necessary exercise to understand PyMC3 inner workings and other sampling algorithms (HMC, NUTS, ...)
