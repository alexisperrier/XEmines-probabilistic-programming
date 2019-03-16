# Titanic PyMC3

# Intro
The titanic dataset is a great stating dataset as it offers several types of features and well-known features engineering problems. It is a case of binary classification predicting the survival chance of several hundred passengers.
It is easy to reach high AUC with standard machine learning approach. see Kaggle.
Here we will apply probabilistic programming approach to the problem. Goal is to see what PP can bring that scikit ML cannot. In terms of interpretation and understanding of the model and the predictions.

# The steps

The steps of a simple probabilistic programming modelisation are
1. variable modelisation
    1. Determine the distribution of the input variables
    2. Determine the distribution of the target variable
    3. Determine the relation between the target and the input

2. Infer the target distribution by sampling the a posteriori distribution

3. Interpret, rinse and repeat

# Starting simple
"Women and children first" as the saying goes. And in fact the data shows that women and children have a higher survival rate than adult men. The data also shows that money may not bring happiness but it sure helps to travel first class if your ship is sinking.
## Input variables

So we will start by limiting ourselves to 3 input variables: age, gender and class.
Gender is a binary variable and will be modeled as a Bernoulli distribution with p = Nw / (Nw + Nm).
Class is a 3 state multinomial variable and should be modeled as a Categorical distribution with p_i = N_i / (N_1 + N_2 + N_3).
Finally Age is a continuous variable with values between 0 - 100. We will model it as a Uniform distribution, thus avoiding any prior assumptions on the age distribution. The age variable has many missing value and we will for the moment simply fill in these missing values by the average for each gender - class subset.

In pymc we write

    code

Before moving on to the next step, let's check that the modelisation makes sense.
We generate samples from the distribution of the variables with ```pm.sample()```. Here we choose one of the samplers available. At this stage it does not really matter which one we choose. Let's go for Metropolis.

We can check that the distributions we have chosen are correct by looking at the trace object.
The categorical ratios are close for gender and class. The distribution for Age is obviously wrong if we compare the histogram of the real age values with the ones from the generated distribution. We'll keep that Uniform distribution for now.

We also need to model the output variable. Since it's a binary variable, we'll also use the Bernoulli distribution with parameter p. However in this case p is the parameter we want to infer from the data. It is an unknown stochastic variable.

    yhat = pm.Bernoulli('yhat', p, observed = df.survived )

So we need to decide on the relation between our unknown variable p and our input data. Since we are in a binary classification context with p being a probability (likelihood ?) we can simply write p as

    code

Things I don't get:

* Why do I get a matrix for p (4 * 1000, n_samples)

Because for each passenger we get 4*1000 samples of the posteriori distribution

So each passenger probability of survival is the mean of all 4*1000 samples

* how to I visualize that the result p make sense

* Why is the model not working ?
all p are ~1










<!--  -->
