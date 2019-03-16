# Jour 3: A/B testing + sampling: NUTS, Metropolis, HMC

Application du MCMC
- start with Gaussian mix in  Ch3_IntroMCMC_PyMC3.ipynb
plus de details

A  Conceptual  Introduction  to Hamiltonian  Monte  Carlo Michael Betancourt 1701.02434.pdf

Gibbs sampling and HMC in
An Introduction to Probabilistic Programming 1809.10756.pdf CH 3

see https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm
Gibbs sampling, involves choosing a new sample for each dimension separately from the others, rather than choosing a sample for all dimensions at once.

see also http://people.duke.edu/~ccc14/sta-663-bootstrap/MCMC.html
Advantages of Gibbs sampling

No need to tune proposal distribution
Proposals are always accepted
Disadvantages of Gibbs sampling

Need to be able to derive conditional probability distributions
need to be able to draw random samples from contitional probability distributions
Can be very slow if paramters are correlated becauce you cannot take “diagonal” steps (draw picture to illustrate)


Diagnosing Convergence

* Using `MAP` to improve convergence
see also Probabilistic programming in Pythonusing PyMC3.pdf

* burn-in period

- Metropolis, Slice sampling, or the No-U-Turn Sampler (NUTS), HamiltonianMC

Metropolis:
https://www.quantstart.com/articles/Markov-Chain-Monte-Carlo-for-Bayesian-Inference-The-Metropolis-Algorithm

* Chris Fonnebeck video
https://youtu.be/5TyvJ6jXHYE?t=1188
1. MCMC and Metropolis - Random Walk based sampling
2. Hamiltonian MC
3. NUTS

# HMC Hamiltonian Monte Carlo explained
https://arogozhnikov.github.io/2016/12/19/markov_chain_monte_carlo.html
visual demo et some math



Varying intercept model = ADVI https://youtu.be/5TyvJ6jXHYE?t=1295
Kullback Leibler divergence
sampling becomes optimization
can't get to posterior distribution directly
minimizing ELBO <=> max KL distance
ADVI: Automatic Differentiation Variational Inference

# A/B testing
AB testing example in Ch2_MorePyMC_PyMC3
http://localhost:8888/notebooks/day_02/Ch2_MorePyMC_PyMC3.ipynb#Example:-Bayesian-A/B-testing

see also
Bayesian Poisson A/B Testing in PYMC3 on Python
http://barnesanalytics.com/bayesian-poisson-ab-testing-in-pymc3-on-python


A/B Testing with Hierarchical Models in Python
https://blog.dominodatalab.com/ab-testing-with-hierarchical-models-in-python/

see Data generation
http://localhost:8888/notebooks/day_02/Ch2_MorePyMC_PyMC3.ipynb#Modeling-approaches

Phase detection in time series
- IoT, simple texting example
- application to ELX sales, financial series, ...

Etant donnée une serie temporelle, nous allons detecter si un changement de phase apparait et determiner les parametres des differentes phases.

- Etude d'un cas d'école: nombre de messages émis par jour sur une periode donnée.
- Modelisation du phénomène, resultats, interpretation

- the maximum a posteriori (MAP)
- fonction de cout exotique, proprietaires et adaptées.

TD: Appliquer les methodes sur les series temporelles de type demand planning, stock market, etc .... et a géneraliser a N le nombre de phases à detecter. Les etudiants presenteront en fin de journée leur resultats
Posterior analysis

datasets:
- electrolux demand planning
- coal mining disaster
https://docs.pymc.io/notebooks/getting_started#Case-study-2:-Coal-mining-disasters
- S&P 500
https://docs.pymc.io/notebooks/getting_started#Case-study-1:-Stochastic-volatility


# MAP
https://docs.pymc.io/notebooks/getting_started#Maximum-a-posteriori-methods


# specific loss functions

# Choosing prior
Prior Choice Recommendations

https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations


* Flat prior;
* Super-vague but proper prior: normal(0, 1e6);
* Weakly informative prior, very weak: normal(0, 10);
* Generic weakly informative prior: normal(0, 1);
* Specific informative prior: normal(0.4, 0.2) or whatever. Sometimes this can be expressed as a scaling followed by a generic prior: theta = 0.4 + 0.2*z; z ~ normal(0, 1);


see Also Kruschke paper
"Even though the prior distribution is often selected to be noncommittal, this does not imply that the prior distribution is an inconvenient nuisance for which a researcher must apologize."
"a well-informed prior distribution can provide inferential leverage."
