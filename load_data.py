"""CIFAR10 small images classification dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.datasets.cifar import load_batch
from keras import backend as K
import numpy as np
import os

# -*- coding: utf-8 -*-
"""Utilities common to CIFAR10 and CIFAR100 datasets.
"""
import sys
from six.moves import cPickle

import threadpool
import time

data = []
labels = []

def load_train_batch(fpath,label_key='labels'):
    """Internal utility for parsing CIFAR data.

    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.

    # Returns
        A tuple `(data, labels)`.
    """
    global data
    global labels
    with open(fpath, 'rb') as f:
        if sys.version_info < (3,):
            d = cPickle.load(f)
        else:
            d = cPickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data.append(d['data'].reshape(d['data'].shape[0], 3, 32, 32))
    labels.append(d[label_key])
def load_test_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.

    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.

    # Returns
        A tuple `(data, labels)`.
    """
    with open(fpath, 'rb') as f:
        if sys.version_info < (3,):
            d = cPickle.load(f)
        else:
            d = cPickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels
def load_data():
    """Loads CIFAR10 dataset.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = 'cifar-10-batches-py'
    origin = "./data/"
    path = origin + dirname

    #多线程并行读取训练数据
    fpath_list = []
    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        fpath_list.append(fpath)
    start = time.time()
    pool = threadpool.ThreadPool(10)
    requests = threadpool.makeRequests(load_train_batch, fpath_list)
    [pool.putRequest(req) for req in requests]
    pool.wait()
    end = time.time()
    print("多线程池化读取数据过程花费时间：",end-start)


    global data
    global labels
    x_train = np.array(data).reshape(-1,3,32,32)
    y_train = np.array(labels).reshape(-1)
    fpath = os.path.join(path,'test_batch')
    x_test, y_test = load_test_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)
