# Day 01: motivation, intro, coin flip,

Introduction au PP et Inference Bayesienne. Le pourquoi, le comment et les outils.
Centrées sur des exemples de questions difficiles a resoudre avec les outils ML classique en supervisé ou non supervisé.


## Probabilistic programming

## Les 2 approches

- ML et stats

- black-box vs modele

- frequentist vs Bayes

“Bayesian inference is simply updating your beliefs after considering new evidence.”

“ Frequentists, who ascribe to the more classical version of statistics, assume that probability is the long-run frequency of events (hence the name). For example, the probability of plane accidents under a frequentist philosophy is interpreted as the long-term frequency of plane accidents”

=> frequentist retourne un nombre, une estimation
=> Bayesian retourne une proba

Analogie du code qui a des bugs.

- Mon code a souvent des bugs [prior]
- Mon code passe tout les tests [evidence]

Est ce que mon code est sans bugs?
Reponses:
=> frequentists: oui
=> bayesian: probablement oui (p = 0.8 par exemple)
=> ML: besoin d'une base de codes, leur performances face au tests et s'ils ont des bugs au final ou non


- supervisé vs non supervisé

- les hypothèses
Souvent non respectées mais ca marche quand meme!
Approche frequentist <=> pour un nombre infini d'echantillons
theoreme des grands nombres
E[f] = 1/n \sum f()

d'un point de vue business

- Quelles sont les questions avec reponses

- les questions sans reponses
    - intervalle de confiance
    - phase detection

- pour une question donnée comment y repondre avec ML, stats et PP ?
bon exercice

## Probabilistic programming

- motivation
    - Obtenir une distribution des variables estimées
    - Utiliser nos connaissances a-prori des variables aleatoires présentes
    - définir des fonctions de cout adaptées au probleme
    - moins dependent du ground truth training set.
    - no need for a lot of data

- historique
A Short History of Markov Chain Monte Carlo


# Bayes rules et vocabulaire

# Distribution
A partir de quelques échantillons, comment obtenir une distribution ?
- le bootstrap
- la modelisation: coin flip + bernoulli
[Bayesian Method for hackers](1.2.1 Example: Mandatory Coin-Flip)
How the distribution evolves as more samples are gathered.
~[Coin flip](./ch01_001_coinflip_posterior_proba_evolution.png)



# PP / ML flow
Comparer le workflow en probabilistic programming avec un workflow ML
=> necessité d'avoir le ground truth
- deterministic variables
- random variables
- parameter modelisation


## PP
PP = Bayes + Echantillonage


**PM**
Prise en main des outils, installation des libraries, structure d'un script PyMC3,
exercices simples de modelisation (coin flip), choix des distributions prior, visualisation, interpretation,



**[TD]** coin flip example
