import numpy as np
import pandas as pd

p_u = 1/2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sample_period(t, p_u, n_lmbda,n_ws, param_d, n_samples, T_int, non_stat_latent=False, proxy_non_stat_latent=False,proxy_shift=0.5):
    print("Sampling:{}".format((t, p_u, n_lmbda, n_samples, T_int)))
    column_names = []
    column_names.append('Date')
    l_sigma = 1
    param_as = [1]*n_lmbda
    b = 0.5
    param_cs = [1 + nl for nl in range(0,n_ws)]
    #param_ds = [1 + param_d for nl in range(0,n_ws)]

    mu_noise = 0
    sigma_noise = 1

    if t >= T_int:
        if non_stat_latent:
            p_u = 1
    U = np.random.binomial(1, n_samples*[p_u])*2
    column_names.append('U')
    lmbdas = []
    ws = []
    zs = []
    for i in range(n_lmbda):
        # Latent Lambda_i
        lmbda = np.random.normal(loc=i+1,scale=l_sigma,size=n_samples) #  - t*t*0.1
        lmbdas.append(lmbda)
        column_names.append('lambda{}'.format(i))
        for j in range(n_ws):
            # Observed proxy W_i
            epsilon_noise = np.random.normal(loc=mu_noise, scale=sigma_noise, size=n_samples)
            if proxy_non_stat_latent:
                if t >= T_int:
                    epsilon_noise = epsilon_noise + proxy_shift
            w = param_cs[j]*lmbda + epsilon_noise
            ws.append(w)
            column_names.append('l{}w{}'.format(i,j))

    # UnObserved proxy Z_i
    for p_d in [b,b*2]:
        epsilon_noise = np.random.normal(loc=mu_noise, scale=sigma_noise, size=n_samples)
        z = p_d * U + epsilon_noise
        zs.append(z)
        column_names.append('l{}z{}p{}'.format(i, j,p_d))

    # Outcome Y
    epsilon_noise = np.random.normal(loc=mu_noise, scale=sigma_noise, size=n_samples)
    intervention = np.random.binomial(1, np.absolute(sigmoid(lmbdas[0]))) + epsilon_noise
    epsilon_noise = np.random.normal(loc=mu_noise, scale=sigma_noise, size=n_samples)
    Y_0 = np.array(param_as) @ np.array(lmbdas) + b * U + epsilon_noise
    Y_1 = Y_0 + 2 * intervention
    y = Y_0
    if t >= T_int:
        y = Y_1
    column_names.append('y')
    column_names.append('Y_0')
    column_names.append('Y_1')


    a = np.sum(param_as)
    b = b
    c = np.sum(param_cs)
    d = param_d

    print(f"(a,b,c,d)={(a,b,c,d)}")

    period = np.transpose(np.vstack((np.ones(n_samples)*t, U, *lmbdas, *ws, *zs, y, Y_0, Y_1)))
    period = pd.DataFrame(period)
    period.columns = column_names
    return period, (a,b,c,d)






