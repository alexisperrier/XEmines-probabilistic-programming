# Day 01

- 2 schools: Stats vs ML
    - stats : p-value, small, medium data, hypotheses, hypothesis testing, white box understanding,
        valid at Infinite (law large numbers)
    - ML: large data (>100), black box, hypotheses, optimize loss function, point estimate
    => need 3rd way :

    - Probabilistic programming: small, medium data, no assumptions, white box understanding
        + specific loss functions, modelisation adaptées aux données par le choix de l'a priori
        PP = Bayes + Sampling
        ecosystem
        recent boost with gradient based libraries: Theano, TF proba, ...

    - Use case Quantopian
        - ML: pas adapté
        - Stats: Pas réalistes
        - PP: Obtenir une probabilité effective

    - en pratique:
        - Modeliser l'a priori,
        - définir la relation de vraisemblance,
        - lier la vraisemblance aux donnees observées
        - echantilloner l'a-posteriori

- Bayes Theorem / Laplace
    - Posteriori = priori * vraisemblance / marginal
        - a priori: modélisation du savoir
        - vraisemblance : les donnees
        - posteriori: la connaissance recherchée
        - marginal : ?
    - exercice test medical

- lois conjuguées
    - cas classique: a priori conjugué, on connait la forme analytique de l'a priori et de l'a posteriori
    - Exemples : Binomial Beta, Gaussien / Gaussien, Multinomial Dirichlet

    - sampling necessaire: quand la distribution a priori n'a pas de conjugué.


- pratique 0: PyMC3 intro
    - install, theano
    - code structure, workflow
    -

- pratique 1: Gaussienne
    - generer une gaussienne, poisson, beta, ... avec statsmodel et PyMC3
    - generer une gaussienne: retrouver les parametres de la gaussienne
        - summary, plots, ...
        - faire varier le nombre de data, le nombre d'echantillons

- pratique 2: pile ou face, dice
    - generer donnees avec p != 0.5
    - retrouver p avec Uniform, beta, beta params as uniform
    - dice

- TD: polling
