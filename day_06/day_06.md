Outlier Detection via Markov Chain Monte Carlo
https://bugra.github.io/work/notes/2014-04-26/outlier-detection-markov-chain-monte-carlo-via-pymc/


# LGD
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
