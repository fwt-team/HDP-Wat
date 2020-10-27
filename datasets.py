# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: datasets.py
@Time: 2020-03-17 22:17
@Desc: datasets.py
"""


def split2group(data, group):

    len, wid, ch = data.shape
    batch_len = len // group
    batch_wid = wid // group

    datas = []
    for i in range(group):
        for j in range(group):
            if i == group - 1:
                if j == group - 1:
                    datas.append(data[i * batch_len:, j * batch_wid:, :])
                else:
                    datas.append(data[i * batch_len: (i + 1) * batch_len, j * batch_wid: (j + 1) * batch_wid, :])
            else:
                if j == group - 1:
                    datas.append(data[i * batch_len: (i + 1) * batch_len, j * batch_wid:, :])
                else:
                    datas.append(data[i * batch_len: (i + 1) * batch_len, j * batch_wid: (j + 1) * batch_wid, :])
    datas = [item.reshape((-1, 3)) for item in datas]
    return datas
