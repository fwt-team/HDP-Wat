# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: train_synthetic.py
@Time: 2020-03-16 22:47
@Desc: train_synthetic.py
"""
try:
    import argparse
    import numpy as np

    from model import VIModel_PY, VIModel_DP
    from scipy import io as scio

    from config import DATA_PARAMS, SYNC_DIR
    from utils import console_log

except ImportError as e:
    print(e)
    raise ImportError


class Trainer:

    def __init__(self, args):

        if int(args.algorithm_category) == 0:
            self.model = VIModel_DP(args)
        elif int(args.algorithm_category) == 1:
            self.model = VIModel_PY(args)

    def train(self, data):

        self.model.fit(data)


if __name__ == "__main__":

    # set args
    parser = argparse.ArgumentParser(prog='HIN-datas',
                                    description='Hierarchical Dirichlet process Mixture Models of datas Distributions')
    parser.add_argument('-c', '--algorithm_category', dest='algorithm_category',
                        help='choose VIModel_DP:0 or VIModel_PY:1',
                        default=1)
    parser.add_argument('-name', '--data_name', dest='data_name', help='data_name', default='syn_data1')
    parser.add_argument('-lp', '--load_params', dest='load_params', help='load_params', default=1)
    parser.add_argument('-verbose', '--verbose', dest='verbose', help='verbose', default=1)
    # hyper parameters
    parser.add_argument('-k', '--K', dest='K', help='truncation level K', default=10)
    parser.add_argument('-t', '--T', dest='T', help='truncation level T', default=5)
    parser.add_argument('-z', '--zeta', dest='zeta', help='zeta', default=0.02)
    parser.add_argument('-u', '--u', dest='u', help='u', default=0.9)
    parser.add_argument('-v', '--v', dest='v', help='v', default=0.01)
    parser.add_argument('-tau', '--tau', dest='tau', help='top stick tau', default=1)
    parser.add_argument('-gamma', '--gamma', dest='gamma', help='second stick gamma', default=1)
    parser.add_argument('-th', '--threshold', dest='threshold', help='second threshold', default=1e-7)
    parser.add_argument('-mth', '--mix_threshold', dest='mix_threshold', help='mix_threshold', default=0.01)
    parser.add_argument('-sm', '--second_max_iter', dest='second_max_iter',
                        help='second max iteration of variational inference', default=500)
    parser.add_argument('-m', '--max_iter', dest='max_iter', help='max iteration of variational inference', default=10)
    args = parser.parse_args()

    data = scio.loadmat('{}/{}.mat'.format(SYNC_DIR, args.data_name))
    labels = data['z'].reshape(-1)
    data = data['data'][0].tolist()

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
        args.max_iter = max_iter
        args.tau = tau
        args.gamma = gamma
        args.zeta = zeta
        args.u = u
        args.v = v

    if args.algorithm_category == 1:
        args.omega = 0.5
        args.eta = 0.1

    if args.algorithm_category > 1:
        args.batch_size = 20

    trainer = Trainer(args)
    trainer.train(data)
    pred = trainer.model.predict(data)
    category = np.unique(np.array(pred))
    console_log(pred, labels=labels, model_name='===========hdp-wmm')
