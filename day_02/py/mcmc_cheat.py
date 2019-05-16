true_mu = 10
data = np.random.randn(20) + true_mu

# Initialize
mu_current      = 0.0
proposal_width  = 0.5   # SD of N(mu_current, proposal_width ) used to suggest new mu
trace = [mu_current]    # Memorization of accepted values of mu

for i in range(1000):
    # consider new point: sample from of N(mu_c,p_w)
    mu_proposal = norm(mu_current, proposal_width).rvs()

    # Likelihood:
    # change the likelihood from  norm(mu_current, 1.0).pdf(data).prod()
    # to  np.log(norm(mu_current, 1.0).pdf(data).prod() + 1)
    # to np.log( norm(mu_current, 1.0).pdf(data)  ).sum()
    # L_c =  np.log(norm(mu_current, 1.0).pdf(data).prod() + 1)
    # L_p =  np.log(norm(mu_proposal, 1.0).pdf(data).prod() + 1)
    # L_c =  norm(mu_current, 1.0).pdf(data).prod()
    # L_p =  norm(mu_proposal, 1.0).pdf(data).prod()
    # likelihood_current = norm(mu_current, 1.0).pdf(data).prod()
    # likelihood_proposal = norm(mu_proposal, 1.0).pdf(data).prod()
    likelihood_current =  np.log( norm(mu_current, 1.0).pdf(data)  ).sum()
    likelihood_proposal =  np.log( norm(mu_proposal, 1.0).pdf(data)  ).sum()

    # prior
    # change prior from norm(0, 1).pdf(mu_current)
    # to norm(np.mean(data), 1).pdf(mu_current)
    # talk about MAP as np.mean(data)
    prior_current  = norm(0, 1).pdf(mu_current)
    prior_proposal = norm(0, 1).pdf(mu_proposal)

    # Ncalculate ominator of Bayes formula
    p_current  = likelihood_current * prior_current
    p_proposal = likelihood_proposal * prior_proposal

    ratio = p_current /  p_proposal
    # change the acception criteria from  np.random.rand()  to  np.random.rand() + 1
    accept = (np.random.rand()) > ratio
    # ratios.append(ratio)
    if accept:
        trace.append(mu_current)
        mu_current = mu_proposal
    # else:
    #     rejected.append(mu_proposal)
