# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: config.py
@Time: 2020-02-15 00:05
@Desc: config.py
"""

import os

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

# Local directory for datasets
DATASETS_DIR = os.path.join(REPO_DIR, 'datas')

SYNC_DIR = os.path.join(DATASETS_DIR, 'synthetic')

RESULT_DIR = os.path.join(REPO_DIR, 'result')

# difference datasets config
# K, T, mix_threshold, max_iter, second_max_iter, threshold, group, dim, max_hy1f1_iter, tau, gamma, zeta, u, v

DATA_PARAMS = {

    'syn_data1': {
        0: (10, 5, 0.01, 20, 1000, 1e-7, 4, 4, 5000, 1, 0.01, 1, 1, 0.01),
        1: (10, 10, 0.01, 20, 1500, 1e-7, 4, 4, 5000, 1, 0.01, 1, 1, 0.01),
    },

    'nyu': {
        0: (20, 5, 0.01, 10, 1000, 1e-7, 5, 3, 5000, 20, 0.01, 0.1, 0.1, 0.01),
        1: (20, 10, 0.01, 10, 1000, 1e-7, 5, 3, 5000, 20, 0.01, 0.1, 0.1, 0.01),
    },

    'yeast': {
        0: (20, 20, 0.01, 10, 500, 1e-7, 5, 7, 5000, 20, 0.01, 0.1, 1, 0.01),
        1: (20, 20, 0.01, 10, 500, 1e-7, 5, 7, 5000, 20, 0.01, 0.1, 1, 0.01),
    },
    'Spellman': {
        0: (20, 20, 0.01, 10, 500, 1e-7, 5, 17, 5000, 20, 0.01, 0.1, 1, 0.01),
        1: (20, 20, 0.01, 10, 500, 1e-7, 5, 17, 5000, 20, 0.01, 0.1, 1, 0.01),
    },
    'Sporulation': {
        0: (20, 20, 0.01, 10, 800, 1e-7, 5, 7, 5000, 20, 0.01, 0.1, 1, 0.01),
        1: (20, 20, 0.01, 10, 500, 1e-7, 5, 7, 5000, 20, 0.01, 0.1, 1, 0.01),
    },
    'Human_Fibroblasts': {
        0: (20, 20, 0.01, 10, 500, 1e-7, 5, 17, 5000, 20, 0.01, 0.1, 1, 0.01),
        1: (20, 20, 0.01, 10, 500, 1e-7, 5, 17, 5000, 20, 0.01, 0.1, 1, 0.01),
    },
}
