# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: train_gene.py
@Time: 2020/7/1 下午2:17
@Desc: train_gene.py
"""
try:
    import argparse
    import numpy as np
    import time

    from scipy import io as scio
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

    from model import VIModel_DP, VIModel_PY
    from config import DATA_PARAMS, RESULT_DIR
    from utils import console_log, s_kmeans

except ImportError as e:
    print(e)
    raise ImportError


class Trainer:

    def __init__(self, args):

        if int(args.algorithm_category) == 0:
            self.model = VIModel_DP(args)
        elif int(args.algorithm_category) == 1:
            self.model = VIModel_PY(args)
        else:
            pass

    def train(self, data):

        self.model.fit(data)


if __name__ == "__main__":

    # set args
    parser = argparse.ArgumentParser(prog='HIN-datas',
                                    description='Hierarchical Dirichlet process Mixture Models of datas Distributions')
    parser.add_argument('-c', '--algorithm_category', dest='algorithm_category',
                        help='choose VIModel_DP:0 or VIModel_PY:1',
                        default=1)
    parser.add_argument('-name', '--data_name', dest='data_name', help='data_name', default='Sporulation')
    parser.add_argument('-lp', '--load_params', dest='load_params', help='load_params', default=1)
    parser.add_argument('-verbose', '--verbose', dest='verbose', help='verbose', default=1)
    # hyper parameters
    parser.add_argument('-k', '--K', dest='K', help='truncation level K', default=6)
    parser.add_argument('-t', '--T', dest='T', help='truncation level T', default=50)
    parser.add_argument('-z', '--zeta', dest='zeta', help='zeta', default=0.02)
    parser.add_argument('-u', '--u', dest='u', help='u', default=0.9)
    parser.add_argument('-v', '--v', dest='v', help='v', default=0.01)
    parser.add_argument('-tau', '--tau', dest='tau', help='top stick tau', default=1)
    parser.add_argument('-gamma', '--gamma', dest='gamma', help='second stick gamma', default=1)
    parser.add_argument('-th', '--threshold', dest='threshold', help='second threshold', default=1e-7)
    parser.add_argument('-mth', '--mix_threshold', dest='mix_threshold', help='mix_threshold', default=0.05)
    parser.add_argument('-sm', '--second_max_iter', dest='second_max_iter',
                        help='second max iteration of variational inference', default=-1)
    parser.add_argument('-m', '--max_iter', dest='max_iter', help='max iteration of variational inference', default=100)
    args = parser.parse_args()

    data = scio.loadmat('./datas/gene/{}.mat'.format(args.data_name))['data']
    time_data = data
    # data = 2 ** data
    data = data / np.linalg.norm(data, axis=1, keepdims=True)

    print('begin training......')
    print('========================dataset is {}========================'.format(args.data_name))

    K, T, mix_threshold, max_iter, second_max_iter, threshold, group, dim, \
    max_hy1f1_iter, tau, gamma, zeta, u, v = DATA_PARAMS[
        args.data_name][args.algorithm_category]

    k = group
    pred = KMeans(n_clusters=k).fit_predict(data)
    datas = []
    for i in range(k):
        datas.append(data[pred == i])

    if int(args.load_params) == 1:
        args.K = K
        args.T = T
        args.mix_threshold = mix_threshold
        args.max_hy1f1_iter = max_hy1f1_iter
        args.second_max_iter = second_max_iter
        args.threshold = threshold
        args.tau = tau
        args.gamma = gamma
        args.zeta = zeta
        args.u = u
        args.v = v
        args.gene = True
    if args.algorithm_category == 1:
        args.omega = 0.1
        args.eta = 0.1

    args.max_iter = max_iter
    args.test_data = datas

    trainer = Trainer(args)
    begin = time.time()
    trainer.train(datas)
    end = time.time()
    pred = trainer.model.predict([data])
    print(pred[:500])
    c = trainer.model.xi

    print("time: {}".format(end - begin))
    print(np.unique(np.array(pred)))
    category = np.unique(np.array(pred))
    console_log(pred, data=data, mu=c, model_name='===========hdp-wmm', newJ=len(category))
