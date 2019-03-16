import pandas as pd
import numpy as np
import pymc3 as pm

if __name__ == "__main__":

    # -------------------------------------------------------------
    #  step 1: load and prepare data
    # -------------------------------------------------------------
    # load dataset
    df = pd.read_csv('titanic.csv')[['age','pclass','sex','survived']]
    # binary encoding sex
    df['gender'] = df.sex.apply(lambda d : 1 if d == 'female' else 0 )
    df = df.drop('sex', axis = 1).reset_index(drop = True)
    # age: replace missing values by mean for each sex-pclass subset
    age = df.groupby(by = ['gender','pclass']).mean().reset_index()[['age','pclass','gender']].rename(columns = {'age': 'avg_age'})
    df = df.merge(age, on = ['gender','pclass'], how = 'outer')
    df.loc[df.age.isna(), 'age'] = df[df.age.isna()]['avg_age']
    df = df.drop('avg_age', axis = 1).reset_index(drop = True)

    # -------------------------------------------------------------
    #  step 2: modelisation
    # -------------------------------------------------------------
    ratio_women_men = df.gender.value_counts()[1] / df.shape[0]
    ratio_class = [df.pclass.value_counts()[i] / df.shape[0] for i in pd.Categorical(df.pclass).categories ]

    with pm.Model() as model:
        gender  = pm.Bernoulli('gender', p = ratio_women_men )
        pclass  = pm.Categorical('pclass', p = ratio_class )
        age     = pm.Uniform('age', lower = 0, upper = 100 )

        trace = pm.sample(100, step=pm.Metropolis())


    with pm.Model() as model:
        # gender  = pm.Bernoulli('gender', p = ratio_women_men )
        # pclass  = pm.Categorical('pclass', p = ratio_class )
        # age     = pm.Uniform('age', lower = np.mean(df.age), upper = np.max(df.age) )
        age     = np.mean(df.age) +  pm.HalfNormal('age',  sd = 20 )
        mu      = 1 + pm.math.dot(df.age, age )
        p       = pm.Deterministic( 'p', 1 / ( 1 + pm.math.exp( -mu   )  )   )

        y_obs  = pm.Bernoulli('y_obs', p=p, observed=df['survived'])

        trace = pm.sample(1000, step=pm.NUTS())

    phat   = trace['p'].mean(axis=0)

    pm.traceplot(trace, ['age'])


    mu      = pm.math.dot(df.age, age ) + pm.math.dot(df.pclass, pclass ) +  pm.math.dot(df.gender, gender )
    pm.summary(trace_, ['gender', 'pclass','age'])

    trace['p'].mean(axis=0)
    theta   = trace_['theta'].mean(axis=0)


trace['pclass']
unique, counts = np.unique(trace['pclass'], return_counts=True)


        yhat  = pm.Bernoulli( 'yhat' , theta, observed = y)



        mu    = alpha + pm.math.dot(x,beta)
        theta = pm.Deterministic( 'theta', 1 / ( 1 + pm.math.exp( -mu   )  )   )
        bd = pm.Deterministic('bd' ,-alpha / beta)
        start_ = pm.find_MAP()
        step_ = pm.NUTS()
        trace_ = pm.sample(10000, step_, start_)


trace['pclass']
unique, counts = np.unique(trace['pclass'], return_counts=True)


# --------------------------------------------------
# x = df[["pclass", "age","gender"]]
x = df[["age"]]
x["intercept"] = 1
y = df["survived"]

with pm.Model() as logistic_model:
    effects = pm.Normal('effects', mu=0, tau=2. ** -2, shape=(x.shape[1],1))

    mu      = pm.math.dot(x, effects )
    p       = pm.Deterministic( 'p', 1 / ( 1 + pm.math.exp( -mu   )  )   )

    Y_obs = pm.Bernoulli('Y_obs', p=p, observed=y)

    trace = pm.sample(1000, pm.Metropolis())

trace['p'].mean(axis=0)


# ---------------
x = df[["age"]]
x["intercept"] = 1
y = df["survived"]

with pm.Model() as logistic_model:
    alpha = pm.Normal('alpha', mu = 0, sd = 10)
    # age     = pm.Uniform('age', lower = np.min(df.age), upper =  np.max(df.age) )
    age     = pm.Normal('age', mu=np.mean(df.age), sd = np.std(df.age) )
    mu      = alpha + pm.math.dot(df.age, age )
    p       = pm.Deterministic( 'p', 1 / ( 1 + pm.math.exp( -mu   )  )   )

    Y_obs   = pm.Bernoulli('Y_obs', p=p, observed=y)
    trace   = pm.sample(1000, pm.Metropolis())

trace['p'].mean(axis=0)



# ---------------
