# Titanic PyMC3

# The case for probabilistic programming

Data science is not limited to learning scikit-learn and about overfitting.
One of the main frustration with Data science is the difficulty of linking data and predictive modeling to bring real business value.
One thing is to increase your AUC score by a few percent by meta parameter optimization of XGBoost.
Another is to be able to gather the data, the algorithms and build a data based project to answer business questions.
Questions that often cannot be answered by straightforward machine learning processes but are more in the domain of statistical methods.
cite "the two approaches"
However, statistical methods are often limited to regression techniques if you are not a "real" statisticians. Also bad rap about p-values and p-hacking.
Any way both data science, and statistical approaches suffer from being reliant on assumptions that are not often met in the real worl: sample independance, Gaussian (Normal) distribution, and so forth. Finally predictive analytics and statistical regression give point estimate. A point estimate is the estimated single value of the parameter or the target that is to be inferred. Statistical methods also give confidence intervals but in both cases, you don't have a good understanding of the distribution of the target variable. What is its shape, are its moments, how does it evolve when we add more data or when we model the data generation process differently.

Having the distribution of the target random variable would give us a much better understanding of the outcome.
Do we have a signal or is it just random
Is the inferred value reliable or is it likely to be chaotic


This is where bayesian inference comes in and with it probabilistic programming.
probabilistic programming comes from the ability of writing code with random variables defined with standard distributions. think Poisson, Gaussian, Uniform.
By writing simple relations between the input variables and the outcome. For instance a simple linear relation would come down to
specify endocog
x_0 = N(0,1)
x_1 = N(0,1)

specify exocog as such

y = a x_0 + b x_1







As a data scientist one of the most important thing to do is to keep learning and training yourself to be able to answer
