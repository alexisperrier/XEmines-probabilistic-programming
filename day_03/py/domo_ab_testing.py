import pandas as pd
import numpy as np
import pymc3 as pm
import seaborn as sns
import scipy.stats as stats

def flat_columns(c):
    replace_dict = {" ":"_", "(":"", ")":"","/":"","-":"_","?":"_" }
    c = c.lower()
    for k,v in replace_dict.items():
        c = c.replace(k,v)
    return c

if __name__ == '__main__':

    # load data
    file = '/Users/alexis/amcp/ECU/data/mar11/Ad_Tracking_Chicago_Zip_AJ.csv'
    adf = pd.read_csv(file).dropna()

    adf.columns = [  flat_columns(c) for c in adf.columns  ]

    adf = adf[adf.campaign == 'Chicago'].reset_index()


    file = '/Users/alexis/amcp/ECU/data/mar11/Chicago_DCM_Cost_AJ.csv'
    odf = pd.read_csv(file).dropna()

    odf.columns = [  flat_columns(c) for c in odf.columns  ]

    # by campaign
    cdf = odf.campaign.value_counts()
    campaigns = cdf[cdf > 100].keys()
    df = odf[odf.campaign.isin(campaigns[:2])][['date','campaign','channel_partner', 'company','impressions','clicks','total_conversions']].copy()



    df = df.groupby( by = ['date','campaign'] ).sum().reset_index()
    # pilot vs not pilot
    sns.lineplot(x='date', y='impressions', data = df[df.campaign == campaigns[0]])
    sns.lineplot(x='date', y='impressions', data = df[df.campaign == campaigns[1]])

    pilot  = df[df.campaign == campaigns[0]].impressions.values
    post   = df[df.campaign == campaigns[1]].impressions.values

    # -------------------------------------------------------------------
    #  Estimate conversion 1st campaign: p
    # -------------------------------------------------------------------
    impressions =  df[df.campaign == campaigns[1]].impressions.values
    clicks = df[df.campaign == campaigns[1]].clicks.values
    non_zeros = impressions >0
    impressions = impressions[non_zeros]
    clicks = clicks[non_zeros]


    idx = 1

    data = np.concatenate([np.zeros(impressions[idx] - clicks[idx]), np.ones(clicks[idx])])
    np.random.shuffle(data)

    with pm.Model() as model:
        p = pm.Uniform('p', 0,1)
        y = pm.Bernoulli('y', p, observed = data)
        trace = pm.sample(10000, step = pm.Metropolis() )

    pm.summary(trace)
    pm.traceplot(trace)

    result = pd.DataFrame()

    for idx in range(len(clicks)[10:15]):

        data = np.concatenate([np.zeros(impressions[idx] - clicks[idx]), np.ones(clicks[idx])])
        np.random.shuffle(data)

        with pm.Model() as model:
            p = pm.Uniform('p', 0,1)
            y = pm.Bernoulli('y', p, observed = data)
            trace = pm.sample(10000, step = pm.Metropolis() )

        res = pm.summary(trace)
        res['idx'] = idx
        print(res)
        result = pd.concat( [result, res] )

    # -------------------------------------------------------------------
    #  Estimate conversion 1st campaign: p over all days
    # -------------------------------------------------------------------
    impressions =  df[df.campaign == campaigns[1]].impressions.values
    clicks = df[df.campaign == campaigns[1]].clicks.values
    non_zeros = impressions >0
    impressions = impressions[non_zeros]
    clicks = clicks[non_zeros]

    data = {}
    for idx in range(2):
        d = np.concatenate([np.zeros(impressions[idx] - clicks[idx]), np.ones(clicks[idx])])
        np.random.shuffle(d)
        data[idx] = d

        with pm.Model() as model:
            p = pm.Uniform('p', 0,1, shape = 2)
            y = pm.Bernoulli('y', p, observed = data)
            trace = pm.sample(10000, step = pm.Metropolis() )
# AsTensorError: ('Cannot convert [0 0 0 ... 0 0 0] to TensorType', <class 'numpy.ndarray'>)


    # -----------------------------------------------------------------------
    # Beta distribution
    # -----------------------------------------------------------------------
    idx = 1

    with pm.Model() as model:
        data = pm.Beta('p', impressions[idx] + 1 , impressions[idx] - clicks[idx] + 1)
        y = pm.Normal('y', p, 1, observed = data)
        trace = pm.sample(10000 )


    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------

    impressions =  df[df.campaign == campaigns[0]].impressions.values
    clicks = df[df.campaign == campaigns[0]].clicks.values
    non_zeros = impressions >0
    impressions = impressions[non_zeros]
    clicks = clicks[non_zeros]

    with pm.Model() as model:
        a = pm.Uniform('a',0,100)
        b = pm.Uniform('b',0,100)

        theta   = pm.Beta('theta', a, b, shape = len(clicks))
        bn      = pm.Binomial('bn', p=theta, n=impressions, observed=clicks)
        trace   = pm.sample(10000, step = pm.NUTS() )

    # -----------------------------------------------------------------------
    # pilot vs not pilot
    # -----------------------------------------------------------------------

    impressions = [ df[df.campaign == campaigns[i]].impressions.sum() for i in range(2)]
    clicks = [ df[df.campaign == campaigns[i]].clicks.sum() for i in range(2)]

    with pm.Model() as model:
        a = pm.Uniform('a',0,100)
        b = pm.Uniform('b',0,100)

        theta   = pm.Beta('theta', a, b, shape = len(clicks))
        bn      = pm.Binomial('bn', p=theta, n=impressions, observed=clicks)
        trace   = pm.sample(10000, step = pm.NUTS() )



    qw = pd.DataFrame()
    qw['rate'] =  trace['theta'][:,0]
    qw['Phase'] = 'Oct-Dec 18'

    zx = pd.DataFrame()
    zx['rate'] =  trace['theta'][:,1]
    zx['Phase'] = 'Feb-Mar 19'

    sd = pd.concat([qw,zx])

    ax = sns.barplot(x="Phase", y="rate",  data=sd, order = ['Oct-Dec 18','Feb-Mar 19'])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.grid(axis = 'y')
    plt.title("Conversion Rate")

    plt.tight_layout()
    plt.show()
    plt.savefig('Chicago_Pilot_CnversionRate.png')


    # -----------------------------------------------------------------------
    with pm.Model() as model:
        p_A = pm.Uniform("p_A", 0, 1)
        p_B = pm.Uniform("p_B", 0, 1)

        # Define the deterministic delta function. This is our unknown of interest.
        delta = pm.Deterministic("delta", p_A - p_B)

        # Set of observations, in this case we have two observation datasets.
        obs_A = pm.Bernoulli("obs_A", p_A, observed=x)
        obs_B = pm.Bernoulli("obs_B", p_B, observed=y)

        step = pm.Metropolis()
        trace = pm.sample(100000, step=step)
        burned_trace=trace[1000:]

    # plot posteriors

    fig, axs = plt.subplots(1,1, figsize=(9, 6))
    plt.title("Posterior distributions of $p_A$, $p_B$, and delta unknowns")
    sns.distplot(burned_trace["p_A"], label="posterior of $p_A$")
    plt.vlines(true_p_A, 0, 80, linestyle="--", label="true $p_A$ (unknown)")

    sns.distplot(burned_trace["p_B"], label="posterior of $p_BA$")
    plt.vlines(true_p_B, 0, 80, linestyle="--", label="true $p_B$ (unknown)")

    sns.distplot(burned_trace["delta"], label="posterior of $\delta$")
    plt.vlines(true_p_A - true_p_B, 0, 80, linestyle="--", label="true $\delta$ (unknown)")
    plt.tight_layout()

    plt.legend()

    # majority of Delta > 0 => string probability P_A > P_B

    print("Probability site A is WORSE than site B: %.3f" % \
        np.mean(burned_trace["delta"] < 0))

    print("Probability site A is BETTER than site B: %.3f" % \
        np.mean(burned_trace["delta"] > 0))








# -------------------
