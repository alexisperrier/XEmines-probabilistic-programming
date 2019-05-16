# Bootstrap vs Bayes

Calculating the mean of several numbers is one of the first thing kids learn in elementary school. Little do they know that the simple operation of summing the numbers and dividing the result by the number of numbers also corresponds to finding the maximum likelihood estimate! Taking the average is a statistic and as such has inherent bias and  .
It would be weird for those kids to learn that there are other ways to estimate the true mean of a suite of numbers.


We have some data that follows a Poisson distribution of parameter lambda.
Lambda is the mean of the data.

Given the data let's compare several ways to infer lambda.

* simple mean
* bootstrapping the data to infer the mean
* Using Bayesian inference and MCMC sampling

These estimator will all converge to the same value when the number of samples is large enough.

First generate a small amount of data samples.

Then estimate with np.mean
obviously since we don't have a lot of samples the estimation is far from the reality

Then bootstrap

and finally use pm


so when do we have enough sample to use np.mean instead of more complex inference ?
Is the bootstrap ever better than simple mean?
Does it work with other distributions ?

Meta: Can we model the link between number of samples and mean accuracy ?





<!--  -->
