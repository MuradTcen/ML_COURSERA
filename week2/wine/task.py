#!/usr/bin/env python
# -*-coding: utf-8
from __future__ import division
import numpy as np
import pandas as pd
import operator
import pylab as pl
from math import sqrt
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale


file_data = 'wine.data'


def get_X_y(filename):
    X = pd.read_csv(filename, header=None, usecols=list(range(1, 14)))
    y = pd.read_csv(filename, header=None, usecols=[0]).values.reshape(len(X),)
    return X, y


def get_quality_mark(df):
    return KFold(n=len(df), n_folds=5, shuffle=True, random_state=42)


def get_accuracy_classification(X, y, kf):
    k_means = list()
    for i in range(1, 51):
        kn = KNeighborsClassifier(n_neighbors=i)
        kn.fit(X, y)
        array = cross_val_score(estimator=kn, X=X, y=y,
                                cv=kf, scoring='accuracy')
        m = array.mean()
        k_means.append(m)
    return k_means


def get_max_and_id(lst):
    return max(lst), [i for i, j in enumerate(lst) if max(lst) == j][0]


def main():
    # data = pd.read_csv(file_data, header=None)
    # data.columns = [cols_names]
    # y = data['Class']
    # X = data[1:14]
    # print(data['Alcohol'])
    # print(data['Class'].values)
    # for train, test in kf:
    #     print("%s %s" % (train, test))
    # print(kf
    # print(data.tail())
    # train, test = test_and_train(data, 0.67)
    # print(get_accuracy(test, get_predictions(train, test, 7)))
    # print(data)

    X, y = get_X_y(file_data)
    # print(X)
    kf = get_quality_mark(X)
    k_means = get_accuracy_classification(X, y, kf)
    print(get_max_and_id(k_means))
    # print(k_means)

    X_scale = scale(X)
    # print(X_scale)
    k_means_w_scale_X = get_accuracy_classification(X_scale, y, kf)
    print(get_max_and_id(k_means_w_scale_X))
if __name__ == '__main__':
    main()
