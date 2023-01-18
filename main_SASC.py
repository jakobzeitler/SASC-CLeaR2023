"""
SASC

Certain code snippets taken from https://github.com/claudiashi57/fine-grained-SC


"""


import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.optimize import minimize
import simulation
import matplotlib.pyplot as plt


results_directory = 'results/'

# HELPERS
def plot_sa_results_table(df):
    # Plot SA Table
    cell_text = []
    for row in range(len(table)):
        cell_text.append(table.iloc[row])
    plt.figure(dpi=300)
    plt.axis('off')
    plot_table = plt.table(cellText=cell_text, colLabels=table.columns, loc='center')
    plot_table.auto_set_font_size(False)
    plot_table.set_fontsize(6)
    plot_table.scale(1, 2)
    plt.show()

#This is actually never used, legacy code from Shi et al
def ConvexRegression(X, y):
    p, n = X.shape

    # Objective function
    def f(w):
        return ((np.dot(X, w) - y) ** 2).sum()

    def jac_f(w):
        return (-(2 * ((y - np.dot(X, w)).T).dot(X)))

    # Defining constraints
    def sum_con(w):
        return (np.ones((n)).dot(w) - 1)

    dic_sum_con = {"type": "eq", "fun": sum_con}

    def positive_con(w):
        return w

    dic_positive_con = {"type": "ineq", "fun": positive_con}

    cons = [dic_sum_con, dic_positive_con]

    # Scipy optimization
    result = minimize(f, np.ones(n) / n, jac=jac_f, constraints=cons, method="SLSQP")

    return result

# EXAMPLE 1
def german_reuni(option):
    df = pd.read_csv("data/GermanReunificationGDP.csv")
    df["Date"] = list(range(1960,2004))
    T_int = option['T_int']
    Y = df['data.all.Y']
    df['y'] = Y
    proxy_idxs = range(1,6)
    proxy_columns = list(df.columns[proxy_idxs])
    W = df.iloc[:,proxy_idxs]


    reg = LinearRegression(fit_intercept=False, positive=option['non_neg_cof'])
    reg = reg.fit(W[0:T_int], Y[0:T_int])
    SC = reg.predict(W).reshape(-1, 1)

    # Sensitivyt Analysis
    beta_max = get_biggest_magnitude_coefficient(reg.coef_)

    maxdW_window = option['maxdW_window']
    delta_Ws = [W[proxy][T_int-maxdW_window:T_int].mean() - W[proxy][T_int:T_int+maxdW_window].mean()
                for proxy in proxy_columns]

    delta_Ws = []
    for proxy in proxy_columns:
        pre = W[proxy][T_int-maxdW_window:T_int]
        pre_1 = pre[0:int(len(pre)/2)]
        pre_2 = pre[int(len(pre)/2):]
        pre_mean_diff = np.abs(pre_1.mean() - pre_2.mean())

        post = W[proxy][T_int:T_int+maxdW_window]
        post_1 = post[0:int(len(post)/2)]
        post_2 = post[int(len(post)/2):]
        post_mean_diff = np.abs(post_1.mean() - post_2.mean())

        change = pre_mean_diff - post_mean_diff
        delta_Ws.append(np.abs(change))
        print(f"{proxy} change: {change}")

    delta_W_max = np.abs(max(delta_Ws))
    M_max = np.sum(reg.coef_ != 0)
    SCSA_UB = beta_max * M_max * delta_W_max

    sa_results = {"non_neg_cof": option['non_neg_cof'], 'coefficients': reg.coef_, 'beta_max': round(beta_max, 2),
                  "d_W_max": round(delta_W_max, 2), "M_max": M_max,
                  'SCSA_UB': round(SCSA_UB, 2), 'proxy_columns': proxy_columns}

    return df, SC, sa_results


# EXAMPLE 2
def get_prop99(option):

    # This is a copy and paste from Shi's notebook, looks hacky, but works
    filename = 'data/prop99.csv'

    df = pd.read_csv(filename)
    reader = df
    df = df[df['SubMeasureDesc'] == 'Cigarette Consumption (Pack Sales Per Capita)']
    pivot = df.pivot_table(values='Data_Value', index='LocationDesc', columns=['Year'])
    dfProp99 = pd.DataFrame(pivot.to_records())

    allColumns = dfProp99.columns.values

    states = list(np.unique(dfProp99['LocationDesc']))
    years = np.delete(allColumns, [0])
    caStateKey = 'California'
    states.remove(caStateKey)

    yearStart = 1970
    yearTrainEnd = 1989
    yearTestEnd = 2000

    discard1 = ['Massachusetts', 'Arizona', 'Oregon', 'Florida']
    discard2 = ['Alaska', 'Hawaii', 'Maryland', 'Michigan', 'New Jersey', 'New York', 'Washington']
    abadie = ['Colorado', 'Connecticut', 'Montana', 'Nevada', 'Utah']

    discard = np.concatenate([discard1, discard2])
    states = np.array(states)
    mask = np.isin(states, discard)
    states_clean = np.ma.masked_array(states, mask).compressed()
    otherStates = states_clean

    trainingYears = []
    for i in range(yearStart, yearTrainEnd, 1):
        trainingYears.append(str(i))

    testYears = []
    for i in range(yearTrainEnd, yearTestEnd, 1):
        testYears.append(str(i))

    trainDataDict = {}
    testDataDict = {}
    for key in otherStates:
        series = dfProp99.loc[dfProp99['LocationDesc'] == key]
        trainDataDict.update({key: series[trainingYears].values[0]})
        testDataDict.update({key: series[testYears].values[0]})
    series = dfProp99[dfProp99['LocationDesc'] == caStateKey]
    trainDataDict.update({caStateKey: series[trainingYears].values[0]})
    testDataDict.update({caStateKey: series[testYears].values[0]})

    trainDF = pd.DataFrame(data=trainDataDict)
    testDF = pd.DataFrame(data=testDataDict)

    table = ['California', 'Alabama', 'Arkansas', 'Virginia', 'Wisconsin', 'Wyoming']
    years = ['LocationDesc', '1970', '1971', '1972', '1988', '1989']
    tmp = dfProp99.loc[dfProp99['LocationDesc'].isin(table)]

    def make_full(train, test, index):
        full = np.concatenate([train[index].values, test[index].values])
        return full

    def make_prediction(index, convex=True, return_weight=False):
        full = make_full(trainDF, testDF, index)
        if convex:
            w = ConvexRegression(trainDF[index].values, trainDF['California'].values)
            # import ipdb; ipdb.set_trace()
            res = full @ w.x
            coef = w.x
        else:
            reg = linear_model.LinearRegression(fit_intercept=False)
            reg.fit(trainDF[index].values, trainDF['California'].values)
            res = reg.predict(full)
            coef = res.coef_
        if return_weight == True:
            return (res, coef)
        else:
            return res

    years = np.concatenate([trainingYears, testYears])
    #pred1, weight1 = make_prediction(abadie, return_weight=True)
    #SC = pred1.reshape(-1, 1)
    full = pd.concat([trainDF, testDF])
    data = {"Date":years, "y":trainDF.append(testDF)['California']}
    for proxy in option['proxy_columns']:
        data[proxy] = full[proxy]
    data = pd.DataFrame(data)
    # OLS
    X = trainDF[abadie].values
    y = trainDF["California"].values
    reg = LinearRegression(fit_intercept=False, positive=option['non_neg_cof'])
    reg = reg.fit(X, y)
    SC = reg.predict(make_full(trainDF, testDF, abadie)).reshape(-1,1)

    #Sensitivyt Analysis
    #beta_max = np.max(weight1)
    beta_max = get_biggest_magnitude_coefficient(reg.coef_)


    #for maxdW_window in range(0,10):
    #    delta_Ws = [
    #        trainDF[[proxy]][-maxdW_window:].mean().to_numpy() - testDF[[proxy]][:maxdW_window].mean().to_numpy() for
    #        proxy in abadie]
    #    print(delta_Ws)

    maxdW_window = option['maxdW_window']
    #delta_Ws = [trainDF[[proxy]][-maxdW_window:].mean().to_numpy() - testDF[[proxy]][:maxdW_window].mean().to_numpy() for   proxy in abadie]

    delta_Ws = []
    for proxy in abadie:
        pre = trainDF[[proxy]][-maxdW_window:]
        pre_1 = pre[0:int(len(pre)/2)]
        pre_2 = pre[int(len(pre)/2):]
        pre_mean_diff = np.abs(pre_1.mean().values - pre_2.mean().values)

        post = testDF[[proxy]][:maxdW_window]
        post_1 = post[0:int(len(post)/2)]
        post_2 = post[int(len(post)/2):]
        post_mean_diff = np.abs(post_1.mean().values - post_2.mean().values)

        change = pre_mean_diff-post_mean_diff
        delta_Ws.append(np.abs(change))
        print(f"{proxy} change: {change}")

    delta_W_max = np.abs(max(delta_Ws))[0]
    M_max = len(abadie)
    M_max = np.sum(reg.coef_ != 0)
    SCSA_UB = beta_max * M_max * delta_W_max

    sa_results = {"maxdW_window":option['maxdW_window'],'coefficients':reg.coef_,'beta_max': round(beta_max, 2), "d_W_max": round(delta_W_max, 2), "M_max": M_max,
                  'SCSA_UB': round(SCSA_UB, 2), 'proxy_columns': abadie}

    return data, SC, sa_results

# SIMULATION 1
def get_simulation_data(option):
    # 1. Simulate data
    n_periods = option['n_periods']  # two pre, two post (two pre, so we can do delta W)
    n_lambda = option['n_lambda']
    T_int = int(n_periods / 2)  # First post internvetion stage
    periods = []
    np.random.seed(2)
    non_stat_latent = option['non_stat_latent']
    for t in range(n_periods):
        period, params = simulation.sample_period(t=t, p_u=1 / 2, n_lmbda=n_lambda, n_ws=len(option['proxy_columns']), param_d=option['param_d'],n_samples=1000, T_int=T_int,
                                          non_stat_latent=non_stat_latent, proxy_non_stat_latent=option['proxy_non_stat_latent'], proxy_shift=option['proxy_shift'])
        periods.append(period)
    data = pd.concat([p.mean(axis=0) for p in periods], axis=1).transpose()
    return data, params

def get_simulation_SC(option):
    n_periods = option['n_periods']  # two pre, two post (two pre, so we can do delta W)

    data, params = get_simulation_data(option)

    # 2. Run SC
    outcome_column = ['y']
    proxy_columns = option['proxy_columns']

    pre_period = [0, T_int]
    X = data[proxy_columns].loc[list(range(*pre_period)),:]
    y = data[outcome_column].loc[list(range(*pre_period)),:]
    reg = LinearRegression(fit_intercept=False, positive=option['non_neg_cof'])
    reg = reg.fit(X, y)

    w = ConvexRegression(X.values, y.values[:,0])
    # import ipdb; ipdb.set_trace()
    coef = w.x

    beta_max = get_biggest_magnitude_coefficient(reg.coef_[0])
    #beta_max = np.max(coef)
    maxdW_window = option['maxdW_window']
    delta_Ws = [data[[proxy]].loc[T_int - 1 - maxdW_window:T_int - 1, :].mean().to_numpy() - data[[proxy]].loc[T_int:T_int+maxdW_window, :].mean().to_numpy() for proxy in proxy_columns]
    delta_W_max = np.abs(max(delta_Ws))[0]

    z_proxy = f'l0z0p{option["param_d"]}'
    z_proxies = [z_proxy]
    delta_Zs = [data[[proxy]].loc[T_int - 1 - maxdW_window:T_int - 1, :].mean().to_numpy() - data[[proxy]].loc[  T_int:T_int + maxdW_window, :].mean().to_numpy() for proxy in z_proxies]
    delta_Z_max = np.abs(max(delta_Zs))[0]

    SC = reg.predict(data[proxy_columns])
    #SC = (data[proxy_columns].values @ w.x).reshape(-1,1)
    M_max = len(proxy_columns)
    M_max = np.sum(reg.coef_[0] != 0)
    SCSA_UB = beta_max * M_max * delta_W_max

    plot_SCSA(data, SC, SCSA_UB, option)
    sa_results = {"maxdW_window":option['maxdW_window'], 'coefficients':reg.coef_[0],'beta_max':round(beta_max,2),"d_W_max":round(delta_W_max,2),"d_Z_max":round(delta_Z_max,2),"M_max":M_max,'SCSA_UB':round(SCSA_UB,2), 'proxy_columns': option['proxy_columns']}
    return sa_results, params

# HELPER
def get_biggest_magnitude_coefficient(coefficients):
    pos_max = np.max(coefficients)
    neg_max = np.min(coefficients)
    if pos_max >= abs(neg_max):
        return pos_max
    else:
        return neg_max

def plot_SCSA(data, SC,SCSA_UB, option):
    # Sensitivity Analysis
    SC_UB = SC + SCSA_UB
    neg_SC_UB = SC - SCSA_UB

    bound_band_transparency = 0.15

    plot_negative_bound = False
    if 'plot_negative_bound' in option:
        plot_negative_bound = option['plot_negative_bound']

    timesteps = []
    if "n_periods" in option:
        n_periods = option['n_periods']
        timesteps = list(range(0, n_periods))
    else:
        timesteps = pd.to_numeric(data["Date"]).to_numpy()
    T_int = option['T_int']
    T_int_vertical = timesteps[T_int]

    ax_n = 4
    if 'plot_bias' in option:
        ax_n = 5

    fig, axis = plt.subplots(ax_n, figsize=(6,10))
    ax1 = axis[0]
    ax2 = axis[1]
    ax3 = axis[2]
    for i in range(ax_n):
        axis[i].title.set_size(14)
    from matplotlib.ticker import MaxNLocator
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.set_dpi(500)
    fig.subplots_adjust(hspace=0.4)
    #fig.suptitle(option['title'])

    # Plot Synthetic Control
    ax1.title.set_text('Synthetic Control')
    ax1.plot(timesteps, data[['y']], label="Outcome Y", color='black')
    ax1.plot(timesteps, SC, label='Synthetic Control', linestyle='dashed',   color='orange')
    ax1.plot(timesteps[T_int::], SC_UB[T_int::], label='Synthetic Control ± Bound', linestyle='dashed',  color='green')
    ax1.fill_between(timesteps[T_int::], SC_UB[T_int::].reshape(1,-1)[0], neg_SC_UB[T_int::].reshape(1,-1)[0], color='green', alpha=bound_band_transparency)
    ax1.axvline(T_int_vertical, linestyle='--')
    ax1.legend(loc=2)
    #ax1.axhline(0, linestyle='--')
    if plot_negative_bound:
        ax1.plot(timesteps[T_int::], neg_SC_UB[T_int::], linestyle='dashed', color='green')

    if 'plot_bias' in option:
        fig.subplots_adjust(hspace=0.5)
        ax1.plot(timesteps, data[['Y_0']], label="Untreated Outcome Y(0)", color='black',linestyle='dashed')

        bias = SC - data[['Y_0']]
        ax_bias = axis[4]
        ax_bias.set_xlabel("Time")
        ax_bias.title.set_text("Bias")
        ax_bias.plot(timesteps, bias, label="Bias", color='red')
        ax_bias.axhline(0, linestyle='--')
        ax_bias.axvline(T_int, linestyle='--')
        ax_bias.plot(timesteps[T_int::], [SCSA_UB]*len(timesteps[T_int::]), label="± Bound",  linestyle='dashed',color='green')
        ax_bias.plot(timesteps[T_int::], [-SCSA_UB]*len(timesteps[T_int::]), linestyle='dashed',color='green')
        ax_bias.fill_between(timesteps[T_int::],[SCSA_UB]*len(timesteps[T_int::]), [-SCSA_UB]*len(timesteps[T_int::]), color='green', alpha=bound_band_transparency)

        ax_bias.legend(loc=2)
        #ax1.set_ylim(1,4)
        #ax2.set_ylim(-0.2,2.5)
        #ax3.set_ylim(-0.2,27)
        ax_bias.set_ylim(-.8,.8)
    else:
        1

    # Plot Proxies
    if 'plot_proxies' in option:
        ax5 = axis[3]
        ax5.title.set_text('Proxies and Outcome')
        #ax5.set_ylim(0.8,4.2)
        for prox in option['proxy_columns']:
            label = prox
            if "plot_bias" in option:
                label = 'Observed Proxy X'
            ax5.plot(timesteps, data[[prox]], label=label)
        if "z_proxy_columns" in option:
            for prox in option['z_proxy_columns']:
                label = 'Unobserved Proxy Z'
                ax5.plot(timesteps, data[[prox]], label=label)
        ax5.plot(timesteps, data[['y']], label='Outcome Y', color='black')
        ax5.axvline(T_int_vertical, linestyle='--')
        ax5.legend(loc=2)
        if "plot_bias" not in option:
            ax5.set_xlabel("Time")

    # Plot Effect
    ax2.title.set_text('Average Treatment Effect on Treated (ATT)')
    ax2.plot(timesteps, data[['y']] - SC, label="ATT", color='black')
    average_ATT = (data[['y']] - SC)[T_int::].mean()
    print(f"average ATT:{average_ATT}")
    ax2.plot(timesteps[T_int::], (data[['y']] - SC_UB)[T_int::], label="ATT ± Bound", linestyle='dashed', color='green')
    if plot_negative_bound:
        ax2.plot(timesteps[T_int::], (data[['y']] - neg_SC_UB)[T_int::],  linestyle='dashed',
                 color='green')
        ax2.fill_between(timesteps[T_int::], (data[['y']] - SC_UB)[T_int::].values.reshape(1,-1)[0], (data[['y']] - neg_SC_UB)[T_int::].values.reshape(1,-1)[0], color='green', alpha=bound_band_transparency)

    ax2.legend(loc=2)
    ax2.axvline(T_int_vertical, linestyle='--')
    ax2.axhline(0, linestyle='--')
    # Plot Cummulative Effect
    ax3.title.set_text('Cumulative ATT')
    ax3.plot(timesteps, np.cumsum(data[['y']] - SC), label="Cumulative ATT", color='black')
    ax3.plot(timesteps[T_int::], np.cumsum(data[['y']][T_int::] - SC_UB[T_int::]), label="Cumulative ATT ± Bound",  linestyle='dashed',color='green')
    if plot_negative_bound:
        ax3.plot(timesteps[T_int::], np.cumsum(data[['y']][T_int::] - neg_SC_UB[T_int::]),
                 linestyle='dashed', color='green')
        ax3.fill_between(timesteps[T_int::], np.cumsum(data[['y']][T_int::] - neg_SC_UB[T_int::]).values.reshape(1,-1)[0], np.cumsum(data[['y']][T_int::] - SC_UB[T_int::]).values.reshape(1,-1)[0], color='green', alpha=bound_band_transparency)

    ax3.legend(loc=2)
    ax3.axvline(T_int_vertical, linestyle='--')
    ax3.axhline(0, linestyle='--')

    if "proxy_shift" in option:
        plt.savefig(results_directory + f'simulation{option["proxy_shift"]}.png', bbox_inches='tight')
    else:
        plt.savefig(results_directory + f'example{option["proxy_columns"][0]}.png', bbox_inches='tight')
    plt.show()



# RUN EVALUATION

eval_simulation = 1
prop99 = 1
germanreuni = 1

# --- SIMULATION
if eval_simulation:
    proxy_columns_combinations = [['l0w0','l0w1', 'l0w2']]
    proxy_columns_combinations = [['l0w0']]
    options = []
    n_lambdas = [1]
    for n_periods in [12]:
        for proxy_shift in [0.5,0.1]: #[False, True]:
            for n_lambda in n_lambdas:
                for proxy_columns in proxy_columns_combinations:
                    T_int = int(n_periods / 2)  # First post internvetion stage
                    options.append(
                        {
                            'n_periods': n_periods,
                            'n_lambda': n_lambda,
                            'proxy_columns': proxy_columns,
                            'z_proxy_columns':['l0z0p0.5'],
                            'non_neg_cof': False,
                            'T_int':T_int,
                            #'plot_bias':True
                            'title':"Bounds on Synthetic Control (Simulation)",
                            "plot_proxies": True,
                            'plot_bias':True,
                            'maxdW_window':int(n_periods/2),'plot_negative_bound':True,
                            'non_stat_latent':True, #bias
                            'param_d':0.5,
                            'proxy_non_stat_latent':True,
                            'proxy_shift':proxy_shift
                        }
                    )
    sa_rs = []
    for option in options:
        sa_r, params = get_simulation_SC(option)
        a,b,c,d = params
        sa_r['proxy_columns'] = str(sa_r['proxy_columns'])
        sa_r['a']=a
        sa_r['b']=b
        sa_r['c']=c
        sa_r['d']=d
        sa_rs.append(sa_r)
        1
    table = pd.DataFrame(sa_rs)
    print(table.to_markdown())
    print(table.to_latex(index=False))


# --- PROP99
if prop99:
    abadie = ['Colorado', 'Connecticut', 'Montana', 'Nevada', 'Utah']
    proxies = abadie
    sa_rs = []
    for non_neg_cof in [True]:  # [False, True]:
        for window in [2]:
            option = {"T_int": 18, 'title': "California Prop99", "plot_proxies": True, "proxy_columns": proxies,
                      'non_neg_cof': non_neg_cof, 'maxdW_window':window,'plot_negative_bound':True}

            data, SC, sa_r = get_prop99(option)
            plot_SCSA(data, SC, sa_r['SCSA_UB'], option)
            sa_r['proxy_columns'] = str(sa_r['proxy_columns'])
            sa_rs.append(sa_r)
    table = pd.DataFrame(sa_rs)
    print(table.to_markdown())

# --- GERMAN REUNIFICATION
if germanreuni:
    proxies = ['gdp.USA','gdp.Austria','gdp.Netherlands','gdp.Switzerland','gdp.Japan']
    sa_rs = []
    for non_neg_cof in [True]:  # [False, True]:
        for window in [2]:

            option = {"T_int": 31, 'title': "German Reunification", "plot_proxies": True, "proxy_columns": proxies,
                      'non_neg_cof': non_neg_cof, 'maxdW_window':window,'plot_negative_bound':True}


            data, SC, sa_r = german_reuni(option)
            plot_SCSA(data, SC, sa_r['SCSA_UB'], option)
            sa_r['proxy_columns'] = str(sa_r['proxy_columns'])
            sa_rs.append(sa_r)
    table = pd.DataFrame(sa_rs)
    print(table.to_markdown())

