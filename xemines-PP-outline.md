# Intro

Supervised and unsupervised learning is a tiny portion of AI
cloud visual with  RL, Bayesian inference, tree search, evolution, knowledge graphs
generative adversarial networks (GAN), ...
Super advanced techs on one side, vs companies who are slowly getting up to speed on AI.
But Companies want AI to answer questions.
Questions that are usually answered by stats (LRs).
In between LR and ML comes Bayesian Inference
- more informative informations than point estimates
- better understanding of how the model is generated
- can answer questions that both LR and ML cannot.

# Argument
The course is centered around a simple use case and related dataset.
Credit default in retail transactions.
In general, questions companies are trying to have answers fall into 2 categories:

Predictive analysis
1. how can we best predict that a customer will default on his//her credit? and its corrolary, by how much. Machine Learning and Deep Learning models are the goto methods for building these predictive models.
2. What are the main factors driving defaulting. This is where classic LR shines while ML can also bring answers (feature importance in RF).

PP/Bayes inference is third way between LR and ML that offers a much more thorough understanding of the predicted values ...

Usually the decision to allow a credit fund to a customer must done when the customer applies for a retail credit card.

“The result of a Bayesian analysis is the posterior distribution, not a single value but a distribution of plausible values given the data and our model.” Excerpt From: Osvaldo Martin. “Bayesian Analysis with Python.” iBooks.
=> most probable value = peak of the distribution
=> spread of distribution ~ uncertainty about the value of a parameter

# Course progression

Given a retail credit dataset, the students will be asked to answer a series of business questions on that dataset. The idea is to mix questions that can be answered by straightforward ML modeling and questions that PP can answer best: confidence intervals on the predicted values for instance.

Nous suivrons le livre Bayesian methods for hackers disponible en ligne.

Chaque jour de la semaine sera consacrée a un cas d'application des methodes d'inference bayesienne, aux differentes methodes MCMC (NUTS, ...),

Nous utiliserons la librarie PyMC3


# Jour 1 motivation et intro et coin flip
- Morning motivation
[?=?+?]
Introduction au PP et Inference Bayesienne. Le pourquoi, le comment et les outils.
Centrées sur des exemples de questions difficiles a resoudre avec les outils ML classique en supervisé ou non supervisé.
    - Questions not answered by ML and stats LR => need for a distribution
gives confidence intervals
    - fonction de cout non standard, proprietaires et adaptées.

[PP=?+?]
- Recall the boostrap: from a few samples infer a distribution
Bernoulli p != 0.5 ?
- case: 2 gaussians, not same number of samples, estimate mu and sigma with pymc3
- pymc3 flow vs ML flow
[PP=Bayes+?]
- catching up on Bayes: prior, posterior, likelihood
- pymc = Bayes + sampling (MCMC)

Chap 1 de Bayes methods for hackers.
application  a la determination samplée des distribution des posteriors
[PP=Bayes+Sampling]

**PM**
Tools, libraries
coin flip example

- deterministic variables
- random variables
- parameter modelisation

Prise en main des outils, installation des libraries, structure d'un script PyMC3,
exercices simples de modelisation (coin flip), choix des distributions prior, visualisation, interpretation,


# Jour 2: MCMC et Changement de phase dans une serie temporelle

**AM**
* Markov chains
applications to other fields
* MAP
* Interpretation
* Choosing good  priors: different cases: poisson, gaussian, bernoulli, ...

**PM**
Phase detection in time series
- IoT, simple texting example
- application to ELX sales, financial series, ...

Etant donnée une serie temporelle, nous allons detecter si un changement de phase apparait et determiner les parametres des differentes phases.

- Etude d'un cas d'école: nombre de messages émis par jour sur une periode donnée.
- Modelisation du phénomène, resultats, interpretation

- the maximum a posteriori (MAP)
- fonction de cout exotique, proprietaires et adaptées.


Les etudiants sont appelés a appliquer les methodes du matin sur des series temporelles de type demand planning, stock market, etc .... et a géneraliser a N le nombre de phases à detecter. Les etudiants presenteront en fin de journée leur resultats
Posterior analysis

datasets:
- electrolux demand planning
- coal mining disaster
- S&P 500


# Jour 3: A/B testing + sampling: NUTS, Metropolis, HMC


De meme le matin fera l'objet d'une presentation d'un cas d'application, le AB testing. Nous comparerons les methodes classiques au methodes d'inference bayesienne. Avec pour l'apres midi un TD sur un ou plusieur jeux de données.

- Metropolis, Slice sampling, or the No-U-Turn Sampler (NUTS), HamiltonianMC

# Jour 4: Classification, regression and Theano
Comment appliquer l'inference bayesienne a un cas classique de regression ou de classification.

- iris, titanic ?


- Theano

" Theano is a library that allows expressions to be defined using generalized vector data structures called tensors, which are tightly integrated with the popular NumPy ndarray data structure, and similarly allow for broadcasting and advanced indexing, just as NumPy arrays do. Theano also automatically optimizes the likelihood’s computational graph for speed and provides simple GPU integration."
https://docs.pymc.io/notebooks/getting_started



# Jour 5: supply chain ?

**AM**
- weak / strong prior
* chosing the prior: from non informatiove (flat) priors to weakly informative priors and strong priors. use all the reliable info you have on thr priors.

- prediction generation
- missing values
**PM**
- retail credit case

- supply chain ? https://peadarcoyle.com/2018/12/18/applications-of-bayesian-statistics-supply-chain/

# Concepts


* posterior distribution vs point estimate

* **reporting**: show posterior distribution with mean and std and “Highest Posterior Density (HPD)”
“If we say that the 95% HPD for some analysis is [2-5], we mean that according to our data and model we think the parameter in question is between 2 and 5 with a 0.95 probability. ”
Excerpt From: “Bayesian Analysis with Python.”

* generating predictions: posterior predictive checks. look for differences between the generated data and the actual data informs on ways to improve the model. Which part of teh model can be trusted? mean ok ? rare values ?

# Exercices


# tools & techniques

- theano
-


based on this article
Modelling LGD for unsecured retail loans using Bayesian methods

compare les 2 approches statistique et bayesienne pour predire le LGD (Loss given default), le montant perdu par la banque quand l'emprunteur fait defaut sur son credit.
Comment estimer a la fois la probabilité de non remboursement et le montant a risque dans un cadre de pret bancaire personel (_personal loans that were granted by a large UK bank_).

Dans ce cadre, l'approche statistique va consister a construire 2 modeles,
* une regression logistique pour estimer la probabilité de default
* suivie  par une regression lineaire pour estimer le montant a risque.

Le "score" final est constitué par le produit de la probabilité de  default  fois le montant a risque.

La modelisation bayesienne consiste a estimer une distribution composée de 2 gaussiennes de moyenne et ecart type different suivant la probabilité de default _p_ (Bernoulli). Cette probabilite de default est estimé par une regression logistique.
Si on obtient une prediction de non default (p < seuil), la gaussienne est centrée, sinon elle est estimé par une regression lineaire.
On a donc aussi une approche a 2 etages, 1) estimation de la  probabilité de default puis 2) estimation de la distribution gaussienne du montant a risque.

Les resultats presenté dans l'article montre que les 2 approches obtiennent les memes estimations pour les parametres inconnus (voir table 2) ceci, bien que l'article s'efforce de minimiser la validité de l'approche statistique.

# Resources

- Probabilistic programming in Python using PyMC3
https://peerj.com/articles/cs-55/

Salvatier J, Wiecki TV, Fonnesbeck C. 2016. Probabilistic programming in Python using PyMC3. PeerJ Computer Science 2:e55 https://doi.org/10.7717/peerj-cs.55

- Using Bayesian Decision Making to Optimize Supply Chains
https://twiecki.github.io/blog/2019/01/14/supply_chain/

## PyMC3
- https://github.com/pymc-devs/pymc3
- http://pymc-devs.github.io/pymc3/

# Python/PyMC3 port of the examples in " Statistical Rethinking A Bayesian Course with Examples in R and Stan" by Richard McElreath
https://github.com/aloctavodia/Statistical-Rethinking-with-Python-and-PyMC3

# Getting started with PyMC3
https://docs.pymc.io/notebooks/getting_started
