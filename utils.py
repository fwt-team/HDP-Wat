# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: utils.py
@Time: 2020-02-06 15:10
@Desc: utils.py
"""
import numpy as np
import warnings
import math
from scipy import sparse

from scipy.special import hyp1f1, gammaln
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI, mutual_info_score as MI,\
        adjusted_mutual_info_score as AMI, adjusted_rand_score as AR, silhouette_score as SI, calinski_harabasz_score as CH


def cluster_acc(Y_pred, Y):
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    total = 0
    for i in range(len(ind[0])):
        total += w[ind[0][i], ind[1][i]]
    return total * 1.0 / Y_pred.size, w


def predict(x, mu, k, pi, n_cluster):

    yita_c = np.exp(np.log(pi[np.newaxis, :]) + wmm_pdfs_log(x, k, mu, n_cluster))

    yita = yita_c
    return np.argmax(yita, axis=1)


def wmm_pdfs_log(x, ks, mus, n_cluster):

    WMM = []
    for c in range(n_cluster):
        WMM.append(wmm_pdf_log(x, mus[c:c+1, :], ks[c]).reshape(-1, 1))
    return np.concatenate(WMM, 1)


def wmm_pdf_log(x, mu, k):

    D = x.shape[len(x.shape) - 1]
    pdf = gammaln(D / 2) - (D / 2) * np.log(2 * np.pi) - np.log(hyp1f1(1 / 2, D / 2, k)) + k * (x.dot(mu.T) ** 2)
    return pdf


def calculate_mix(a, b, K):

    lambda_bar = a / (a + b)
    pi = np.zeros(K,)
    for i in range(K):
        temp_temp = 1
        for j in range(i):
            temp = 1 - (lambda_bar[j])
            temp_temp = temp_temp * temp
        pi[i] = lambda_bar[i] * temp_temp
    return pi


def d_hyp1f1(a, b, k, iteration=-1):

    warnings.filterwarnings('ignore')
    result = hyp1f1(a + 1, b + 1, k) / hyp1f1(a, b, k)
    if np.any(np.isnan(result)):
        result = []
        for i in range(k.shape[0]):
            if k[i] < 10:
                tol = 0.01
            else:
                tol = 0.1
            iter = 1
            stack = []
            while True:
                if iter % 2 == 0:
                    temp = (a + iter / 2) / (b + iter - 1) / (b + iter) * k[i]
                else:
                    temp = (a - b - math.floor(iter / 2)) / (b + iter - 1) / (b + iter) * k[i]
                if abs(temp) < tol:
                    iter = iter - 1
                    break
                else:
                    stack.append(temp)

                if iter > (np.inf if iteration == -1 else iteration):
                    break
                iter = iter + 1
            temp = 1
            warnings.filterwarnings('ignore')
            stack = np.vstack((d for d in stack)).reshape(-1)
            while iter > 0:
                temp = 1 + stack[iter - 1] / temp
                iter = iter - 1
            result.append(1 / temp)

        warnings.filterwarnings('ignore')
        result = np.vstack((d for d in result)).reshape(-1)
    return result


def caculate_pi(model, N, T):

    resp = np.zeros((N, T))
    resp[np.arange(N), model] = 1
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    pi = (nk / N)[np.newaxis, :]
    return pi


def log_normalize(v):
    ''' return log(sum(exp(v)))'''
    log_max = 100.0
    if len(v.shape) == 1:
        max_val = np.max(v)
        log_shift = log_max - np.log(len(v)+1.0) - max_val
        tot = np.sum(np.exp(v + log_shift))
        log_norm = np.log(tot) - log_shift
        v = v - log_norm
    else:
        max_val = np.max(v, 1)
        log_shift = log_max - np.log(v.shape[1]+1.0) - max_val
        tot = np.sum(np.exp(v + log_shift[:, np.newaxis]), 1)

        log_norm = np.log(tot) - log_shift
        v = v - log_norm[:, np.newaxis]

    return v, log_norm


def console_log(pred, data=None, labels=None, model_name='cluster', newJ=None, mu=None, verbose=1):

    measure_dict = dict()
    if data is not None:
        measure_dict['si'] = SI(data, pred)
        measure_dict['ch'] = CH(data, pred)
    if labels is not None:
        measure_dict['acc'] = cluster_acc(pred, labels)[0]
        measure_dict['nmi'] = NMI(labels, pred)
        measure_dict['mi'] = MI(labels, pred)
        measure_dict['ar'] = AR(labels, pred)
        measure_dict['ami'] = AMI(labels, pred)
    if mu is not None:
        if len(data.shape) > 2:
            data = np.vstack((i for i in data))
        each_data = [data[pred == i] for i in np.unique(pred)]
        measure_dict['h_ave'] = compute_h_ave(each_data, mu=mu, N=data.shape[0])
        measure_dict['s_ave'] = compute_s_ave(each_data, mu=mu)
    if newJ is not None:
        measure_dict['new_component'] = newJ

    if verbose:
        char = ''
        for (key, value) in measure_dict.items():
            char += '{}: {:.4f} '.format(key, value)
        print('{} {}'.format(model_name, char))

    return measure_dict


def randsample(n, k):

    h = list()
    while (len(h) < k):
        h.append(np.random.randint(0, n-1))

    return h


def s_kmeans(n_components, x, reps=9, maxit=200):

    (N, D) = x.shape
    bestlabel = []
    sumD = np.zeros((n_components))
    bCon = False
    maxit = maxit
    reps = reps
    for t in range(reps):
        mu = x[randsample(N, n_components), :]
        last = 0
        label = np.array([1])
        it = 0
        while np.any(label != last) and it < maxit:
            last = label
            simMat = x.dot(mu.T)
            val = np.max(simMat, 1)
            label = np.argmax(simMat, 1)
            ll = np.unique(label)
            if len(ll) < n_components:
                misscluster = [i for i in range(n_components)]
                misscluster = np.delete(misscluster, ll)
                missNum = len(misscluster)
                idx = np.argsort(val)
                label[idx[:missNum]] = misscluster

            E = sparse.coo_matrix((np.ones(N), ([i for i in range(N)], label)),
                                  shape=(N, n_components))
            mu = E.dot(sparse.spdiags(1 / (np.sum(E, 0) + 1e-10), 0, n_components, n_components)).T.dot(x)
            centernorm = np.sqrt(np.sum(mu ** 2, 1))[:, np.newaxis]
            mu = mu / (centernorm + 1e-10)

            it = it + 1

        if it < maxit:
            bCon = True

        if len(bestlabel) == 0:

            bestlabel = label
            bestcenter = mu
            if reps > 1:
                if np.any(label != last):
                    simMat = x.dot(mu.T)
                D = 1 - simMat
                for k in range(n_components):
                    sumD[k] = np.sum(D[label == k, k], 0)

                bestsumD = sumD
                bestD = D
        else:
            if np.any(label != last):
                simMat = x.dot(mu.T)
            D = 1 - simMat
            for k in range(n_components):
                sumD[k] = np.sum(D[label == k, k], 0)

            if np.sum(sumD) < np.sum(bestsumD):
                bestlabel = label
                bestcenter = mu
                bestsumD = sumD
                bestD = D
    return bestlabel, bestcenter


def compute_h_ave(data, mu, N):
    """
    The homogeneity measures
    data is a list, include j cluster
    :param data:
    :param mu:
    :param N:
    :return: h_ave
    """
    h_ave = 0
    for j, x in enumerate(data):
        ave = (x.dot(mu[j][:, np.newaxis])) ** 2
        h_ave += np.sum(ave)

    h_ave = h_ave / N
    return h_ave


def compute_s_ave(data, mu):
    """
    The separation measures
    data is a list, include j cluster
    :param data:
    :param mu:
    :return: s_ave
    """
    ave = 0
    ave_len = 0
    for i in range(len(data)):
        i_len = data[i].shape[0]
        i_u = mu[i][np.newaxis, :]
        for j in range(len(data)):
            if j == i:
                continue
            else:
                j_len = data[j].shape[0]
                u = i_u.dot(mu[j][:, np.newaxis])[0][0]
                u = min(u, -u)
                ave += i_len * j_len * u
                ave_len += i_len * j_len
    s_ave = ave / ave_len

    return s_ave
