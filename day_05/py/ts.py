import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

if __name__ == "__main__":

    df = pd.read_csv("../data/tree-rings.csv",
        parse_dates = ['year'],
        index_col = 'year',
        infer_datetime_format = True)

    # Aurocorrelation, partial autocorrelation
    from statsmodels.tsa.stattools import acf, pacf
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    acf(df.rings)

    # -----------------------------------------------------
    #  Simple prediction Yhat_t = x_t-1
    # -----------------------------------------------------

    df['pred'] = df.rings.shift(1)

    df.head()

    mape(df.rings, df.pred)

    # -----------------------------------------------------
    #  Rolling
    # -----------------------------------------------------

    df['nowindow'] = df.rings.rolling( 10).mean()
    df['nowindow100'] = df.rings.rolling( 100).mean()
    df['triang'] = df.rings.rolling(wintype = 'triang', 10).mean()
    df['triang100'] = df.rings.rolling(wintype = 'triang', 100).mean()
    fig, ax = plt.subplots(1,1)
    plt.plot(df.rings, alpha = 0.4, label = 'original')
    plt.plot(df.triang, label = 'triang n = 10')
    plt.plot(df.triang100, label = 'triang n = 100')
    plt.show()


    # ---------------------------------------------------------------------
    #  SimpleExpSmoothing yhat_t = alpha * y_t + (1-alpha) * y_t-1
    # ---------------------------------------------------------------------
    from statsmodels.tsa.api import  SimpleExpSmoothing

    fit = SimpleExpSmoothing(df.rings).fit(smoothing_level=0.6)
    df['expfitted'] = fit.fittedvalues
    fit = SimpleExpSmoothing(df.rings).fit(smoothing_level=0.51)
    df['expfitted08'] = fit.fittedvalues

    fig, ax = plt.subplots(1,1)
    plt.plot(df.rings, alpha = 0.4, label = 'original')
    # plt.plot(df.expfitted, label = 'exp smoothing alpha = 0.6', alpha = 0.4)
    plt.plot(df.expfitted08, label = 'exp smoothing alpha = 0.51')
    plt.legend()
    plt.show()


    # ---------------------------------------------------
    #  Seasonal decompose
    # ---------------------------------------------------
    sdf = sm.tsa.seasonal_decompose(df.rings)

    sdf.plot()

    # ------------------------------------------
    # production de lait
    # ------------------------------------------

    df = pd.read_csv('../data/monthly-milk-production-pounds-p.csv')




    # result = sm.tsa.stattools.adfuller(df.rings)
    mlk = sm.tsa.seasonal_decompose(df.milk)

    mlk.plot()

    # ---- Dickey Fuller Test
    result = sm.tsa.stattools.adfuller(df.milk)


    # --------
    from statsmodels.tsa.arima_model import ARIMA

    '''
    Looking at the acf plot of the milk time series shows a p=5 order for the AR modelisation
    We'll compare with AR(2)
    The original milk processus is non stationary
    looking at the differenced process (which is stationnary) we compare with the 2 other
    '''

    modl_AR5 = ARIMA(df.milk, (5,0,0 ) ).fit()
    modl_AR2 = ARIMA(df.milk, (2,0,0 ) ).fit()
    modl_AR2_D1 = ARIMA(df.milk, (2,1,0 ) ).fit()

    plt.plot(modl_AR5.fittedvalues, '.')
    plt.plot(modl_AR2.fittedvalues, '.')
    plt.plot(modl_AR2_D1.fittedvalues, '.')
