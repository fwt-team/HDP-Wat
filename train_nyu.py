# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: train_nyu.py
@Time: 2020/7/1 下午2:17
@Desc: train_nyu.py
"""
try:
    import argparse
    import numpy as np
    import time
    import os
    import re

    from scipy import io as scio
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

    from model import VIModel_DP, VIModel_PY
    from config import DATA_PARAMS, RESULT_DIR
    from utils import console_log
    from datasets import split2group

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


def scalar_data(data, scalar):

    result = data[::scalar, ::scalar, :]
    # return result.reshape((-1, data.shape[2])), result.shape

    return result, result.shape


if __name__ == "__main__":

    # set args
    parser = argparse.ArgumentParser(prog='HIN-datas',
                                    description='Hierarchical Dirichlet process Mixture Models of datas Distributions')
    parser.add_argument('-c', '--algorithm_category', dest='algorithm_category',
                        help='choose VIModel_DP:0 or VIModel_PY:1',
                        default=1)
    parser.add_argument('-name', '--data_name', dest='data_name', help='data_name', default='nyu')
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

    print('begin training......')
    print('========================dataset is {}========================'.format(args.data_name))

    K, T, mix_threshold, max_iter, second_max_iter, threshold, group, dim, \
    max_hy1f1_iter, tau, gamma, zeta, u, v = DATA_PARAMS[
        args.data_name][args.algorithm_category]

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
    if args.algorithm_category == 1:
        args.omega = 0.5
        args.eta = 0.1
    args.max_iter = max_iter

    labels = scio.loadmat('./datas/nyu/nyu_labels.mat')['labels']
    files = os.listdir('./datas/nyu/normals/')
    for name in files:
        index = int(re.sub(r'[a-z]', '', name.split('.')[0]))
        data = scio.loadmat('./datas/nyu/normals/{}'.format(name))['imgNormals']
        label = labels[:, :, index-1:index]

        data, size = scalar_data(data, 3)
        label, _ = scalar_data(label, 3)
        label = label.reshape(-1)

        datas = split2group(data, group)
        trainer = Trainer(args)
        begin = time.time()
        trainer.train(datas)
        c = trainer.model.xi
        end = time.time()
        pred = trainer.model.predict([data.reshape((-1, 3))])

        print(np.unique(np.array(pred)))
        category = np.unique(np.array(pred))
        measure_dict = console_log(pred, data=data.reshape((-1, 3)), labels=label, mu=c,
                                   model_name='===========chpy-wmm', newJ=len(category))

        measure_dict['time'] = (end - begin)
        print('time: {}'.format(measure_dict['time']))

