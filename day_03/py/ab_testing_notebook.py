# AB testing - marketing campaigns

There are several campaigns running to drive traffic on a website.

For each day, we have the number of impressions, clicks and total_conversions
as well as information about the campaign such as channel, partner website, targeted audience, ...
We want to do a posteriori AB testing analysis of the campaigns and find out which performed the best.

We will compare 2 phase of the campaign, a pilot and a post pilot phase that ran successively on the web site.

1. Simple Bernoulli modelisation

We model the conversion as a simple Bernoulli variable with probability p.

The prior p is non informative, and taken as a Uniform(0,1) distribution.


1. Simulation
Let's first simulate the data with a p = 0.01.
Generate 10k samples of the Bernoulli(0.01) distribution with statsmolde

    p_true = 0.05  # remember, this is unknown.
    N = 1500
    occurrences = stats.bernoulli.rvs(p_true, size=N)

    print(occurrences)
    print(np.sum(occurrences))

    # define the prior
    with pm.Model() as model:
        p = pm.Uniform('p', lower = 0, upper = 1)

    # sample the posterior
    with model:
        obs     = pm.Bernoulli("obs", p, observed=occurrences)
        trace   = pm.sample(18000, step=pm.Metropolis())
        trace = trace[1000:]

    fig, ax = plt.subplots(1,1)
    plt.title("Posterior distribution of $p_A$, the true effectiveness of site A")
    sns.distplot(trace["p"])
    plt.vlines(p_true, 0, 90, linestyle="--", label="true $p_A$ (unknown)")
    plt.legend()
    plt.show()


2. Find the p for 1) the pilot and the post pilot, 2) the first and 2nd day of the pilot


2. Beta - Binomial

Now let's consider the Beta(n+1,k+1) distribution to model a coin flip that returned $k$ heads over n trials. Similar to the conversion of a digital ad (clicked / did not click).
Each day / campaign has a different conversion rate, but all these conversion rates are taken from a similar Beta distribution with params a,b unknown. We want to find the a and b.

Infer the posterior for the pilot campaign using a hierarchical model.

    a = ?
    b = ?

    p ~ Beta(a,b)
    observed = Binomial( p, impressions, observed = clicks )


3. constrain a and b.






<!--  -->-
