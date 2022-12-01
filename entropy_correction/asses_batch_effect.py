import torch
import pandas as pd
import statsmodels.api as sm
import numpy as np

from statsmodels.formula.api import ols
from scipy.stats import f as fisher_dist



## corrected_data should be tensor of size (n_features, n_samples)
## Main function assesing the batch effect present in a dataset. This compares the
## F statistic distribution to the fisher distribution.
# def fisher_kldiv_detailed(corrected_data, n_batches, batch_size, batchless_entropy):
#     y = corrected_data
#     length = len(y)
#     y_mean = torch.mean(y, 1).view(length, 1).repeat_interleave(n_batches * batch_size, 1)

#     y_batch_mean = y.view(length, n_batches, batch_size)
#     y_batch_mean = torch.mean(y_batch_mean, 2).repeat_interleave(batch_size, 1)

#     exp_var = torch.sum(torch.square(y_batch_mean - y_mean), 1)
#     unexp_var = torch.sum(torch.square(y - y_batch_mean), 1)

#     N = batch_size * n_batches
#     K = n_batches

#     F_stat = (exp_var/unexp_var) * ((N-K) / (K-1))
#     p = torch.distributions.FisherSnedecor(df1 = K-1, df2 = N-K)
#     log_F = p.log_prob(F_stat)
#     return(log_F - batchless_entropy)

def fisher_kldiv(corrected_data, n_batches, batch_size, batchless_entropy):
    y = corrected_data
    y_dim = y.dim()
    # length = y.size(y_dim-2)
    view_args = []
    for index in range(0, y_dim-1):
        view_args = view_args + [y.size(index)]
    view_args = view_args + [1]
    y_mean = torch.mean(y, y_dim-1).view(*view_args).repeat_interleave(n_batches * batch_size, y_dim-1)

    view_args = []
    for index in range(0, y_dim-1):
        view_args = view_args + [y.size(index)]
    view_args = view_args + [n_batches, batch_size]
    y_batch_mean = y.view(*view_args)
    y_batch_mean = torch.mean(y_batch_mean, y_dim).repeat_interleave(batch_size, y_dim-1)

    exp_var = torch.sum(torch.square(y_batch_mean - y_mean), y_dim-1)
    unexp_var = torch.sum(torch.square(y - y_batch_mean), y_dim-1)

    N = batch_size * n_batches
    K = n_batches

    F_stat = (exp_var/unexp_var) * ((N-K) / (K-1))
    p = torch.distributions.FisherSnedecor(df1 = K-1, df2 = N-K)
    log_F = p.log_prob(F_stat)
    return -(log_F - batchless_entropy)

# def fisher_kldiv(corrected_data, n_batches, batch_size, batchless_entropy):
#     distance = fisher_kldiv_detailed(corrected_data, n_batches, batch_size, batchless_entropy)
#     f_dim = distance.dim()

#     loss_kl = torch.sum(distance, f_dim-1)
#     return loss_kl


def abs_effect_estimate(corrected_data, n_batches, batch_size, batchless_entropy):
    y = corrected_data
    y_dim = y.dim()
    length = y.size(y_dim-2)
    distance = fisher_kldiv(corrected_data, n_batches, batch_size, batchless_entropy)
    f_dim = distance.dim()

    loss_kl = torch.sum(abs(distance), f_dim-1)/length
    return loss_kl




## y is a tensor of size (k, n_batches * batch_size)
def test_batch_effect_fast(y, n_batches, batch_size):
    length = len(y)
    y_mean = torch.mean(y, 1).view(length, 1).repeat_interleave(n_batches * batch_size, 1)

    y_batch_mean = y.view(length, n_batches, batch_size)
    y_batch_mean = torch.mean(y_batch_mean, 2).repeat_interleave(batch_size, 1)

    exp_var = torch.sum(torch.square(y_batch_mean - y_mean), 1)
    unexp_var = torch.sum(torch.square(y - y_batch_mean), 1)

    N = batch_size * n_batches
    K = n_batches

    F_stat = (exp_var/unexp_var) * ((N-K) / (K-1))
    # df2test = 10000
    # p_values = 1 - fisher_dist.cdf(F_stat, dfn = K-1, dfd = df2test)
    p_values = 1 - fisher_dist.cdf(F_stat, dfn = K-1, dfd = N-K)

    return(p_values) 


    

## Functions for testing batch effect
## Slow method for testing batch effect. Here for sanity checks.
def test_batch_effect(y, n_batches, batch_size):
    p_values = list()

    for yy in y:
        d = {'value' : yy, 'batch' : [format(b) for b in range(n_batches) for i in range(batch_size)]}
        dff = pd.DataFrame(data = d)
        model = ols('value ~ batch', data = dff).fit()
        aov_table = sm.stats.anova_lm(model, typ = 2)
        p_value = aov_table.iloc[0,3]
        p_values.append(p_value)

    return(p_values)




def batchless_entropy_estimate(n_batches, batch_size, sample_size = 7000000):
    N = batch_size * n_batches
    K = n_batches
    # df2_test = 10000
    # p = torch.distributions.FisherSnedecor(df1 = K-1, df2 = df2_test)
    # F_stat = np.random.f(K-1, df2_test, sample_size)

    p = torch.distributions.FisherSnedecor(df1 = K-1, df2 = N-K)
    F_stat = np.random.f(K-1, N-K, sample_size)
    log_F = p.log_prob(torch.tensor(F_stat))
    return float(torch.mean(log_F)), float(torch.std(log_F, unbiased = True))


def batchless_entropy_distribution(n_batches, batch_size, minibatch_size, sample_size = 10000):
    N = batch_size * n_batches
    K = n_batches
    minibatch_size = 50
    sample_size = 10000
    p = torch.distributions.FisherSnedecor(df1 = K-1, df2 = N-K)
    F_stat = [numpy.random.f(K-1, N-K, minibatch_size) for index in range(0, sample_size)]
    F_stat = numpy.stack(F_stat)
    log_F = p.log_prob(torch.tensor(F_stat))
    log_F_means = torch.mean(log_F, 1)

    log_F_MEAN, log_F_STD = test.batchless_entropy, test.batchless_entropy_std
    log_dev = log_F_means - log_F_MEAN

    return 
