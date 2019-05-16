import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import seaborn as sns
import pandas as pd

def dpa(y, yhat):
    return (1 - np.sum(np.abs(y - yhat)) / np.sum(yhat) )* 100.0


if __name__ == "__main__":
    df = pd.read_csv('hackathon_demand_planning.csv')
    df.rename(columns = {'product': 'sku'}, inplace = True)
    # combien de produits uniques
    print("# Products {}".format( len(df.sku.unique())   ))

    # choisir un code produit au hasard
    sku = np.random.choice( df.sku.unique() )
    ts = list(df[df.sku == sku].qty)
    print(ts)
    # plot la serie temporelle

    fig, ax = plt.subplots(1,1, figsize = (12,6))
    plt.plot(ts, label = sku)
    plt.show()

    # plot 10 series sur le meme graphe
    fig, ax = plt.subplots(1,1, figsize = (12,6))
    for sku in df.sku.unique()[0:10]:
        ts = list(df[df.sku == sku].qty)
        plt.plot(ts, label = sku)
    plt.legend()
    plt.show()
    plt.tight_layout()

    # correlation des produits
    # flatten les donnees dans un tableau
    data = []
    for sku in df.sku.unique():
        data.append( np.array(df[df.sku == sku].qty)   )
    # data est un tableau de 702 produits de 38 mois

    # matrice de correlation
    mcor = np.corrcoef(data)

    hicor = []
    for i in range(len(data)):
        for j in range(i):
            if mcor[i][j] > 0.8:
                hicor.append( (i,j)   )

    # ---------------------------------------
    #  Baseline
    # ---------------------------------------

    y = df[df.month == '2019_02'].qty.values
    yhat = df[df.month == '2019_01'].qty.values

    print("DPA baseline {:2f}".format( dpa(y,yhat)   ))

    # ---------------------------------------
    #  AR(1)
    # see https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
    # ---------------------------------------

    sku = df.sku.unique()[9]
    ts = df[df.sku == sku].qty.values

    # AR(p) autocorrelation function
    from statsmodels.tsa.stattools import acf
    from statsmodels.graphics.tsaplots import plot_acf

    plot_acf(ts)

    from statsmodels.tsa.arima_model import ARIMA
    from tqdm import tqdm

    y_ar = []
    yhat_ar = []
    for sku in tqdm(df.sku.unique()[120:150]):
        ts = df[df.sku == sku].qty.values
        model = ARIMA(ts[:-1], order=(1,0,0)).fit(disp = 0)
        y_ar.append(ts[-1])
        yhat_ar.append(model.forecast(1)[0][0])

    y_ar = np.array(y_ar)
    yhat_ar = np.array(yhat_ar)
    print("DPA AR(1) {:2f}".format( dpa(y_ar,yhat_ar)   ))


    # ---------------------------------------
    #  pm.AR(1)
    # https://docs.pymc.io/api/distributions/timeseries.html
    # see https://barnesanalytics.com/bayesian-auto-regressive-time-series-analysis-pymc3
    # ---------------------------------------
    import pymc3 as pm
    # choose a random sku

    sku = df.sku.unique()[10]
    ts = df[df.sku == sku].qty.values
    plt.plot(ts)

    with pm.Model() as model:
        k_      = pm.Uniform('k',-1,1)
        tau_    = pm.Gamma('tau',mu=1,sd=1)
        obs     = pm.AR('observed',k=k_,tau_e=tau_,observed=ts[:-1])
        trace   = pm.sample()

    pm.summary(trace)

    # essai avec AR(2)
    with pm.Model() as model:
        k1_     = pm.Uniform('k1',-1,1)
        k2_     = pm.Uniform('k2',-1,1)
        tau_    = pm.Gamma('tau',mu=1,sd=1)
        obs     = pm.AR('observed',rho=[2,k1_,k2_],tau=tau_,observed=ts[:-1])
        trace   = pm.sample()

    pm.summary(trace)


    # -------------------------------------------
    # All skus AR(1)
    # -------------------------------------------

    rt = pd.DataFrame()
    for sku in df.sku.unique()[120:150]:
        ts = df[df.sku == sku].qty.values
        with pm.Model() as model:
            k_      = pm.Uniform('k',-1,1)
            tau_    = pm.Gamma('tau',mu=1,sd=1)
            obs     = pm.AR1('observed',k=k_,tau_e=tau_,observed=ts[:-1])
            trace   = pm.sample(2000, progressbar = False)
            df_tmp = pm.summary(trace, varnames= 'k')
            df_tmp['sku']   = sku
            df_tmp['y']     = ts[-1]
            df_tmp['y-1']   = ts[-2]

            yhat  = ts[-2] * np.mean(trace['k']) + np.mean(trace['tau'])
            df_tmp['yhat']  = yhat
            rt = pd.concat( [rt, df_tmp])
            print("[{}]  y: {} yhat: {:.2f} k {:.2f} dpa {:.2f} ".format(
                sku, ts[-1], yhat, np.mean(trace['k']) , dpa(rt.y,rt.yhat)
            ))

    print("DPA AR(1) {:2f}".format( dpa(rt.y,rt.yhat)   ))




# -----------------------------------------------------
