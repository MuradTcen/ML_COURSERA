import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd


def get_X_y(filename):
    X = pd.read_csv(filename, header=None, usecols=list(range(1, 3)))
    y = pd.read_csv(filename, header=None, usecols=[0]).values.reshape(len(X),)
    return X, y


def get_accuracy_score(test_X, test_y, train_X, train_y):
    clf = Perceptron(random_state=241, shuffle=True)
    clf.fit(train_X, train_y)
    predictions = clf.predict(test_X)
    quality_test = accuracy_score(predictions, test_y)
    return quality_test


def main():
    test_filename = 'perceptron-test.csv'
    train_filename = 'perceptron-train.csv'
    test_X, test_y = get_X_y(test_filename)
    train_X, train_y = get_X_y(train_filename)
    # print(test_X.head())
    # print(test_y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_X)
    X_test_scaled = scaler.transform(test_X)

    second = get_accuracy_score(test_X, test_y, train_X, train_y)
    first = get_accuracy_score(X_test_scaled, test_y, X_train_scaled, train_y)
    res = first - second
    print(first, second, res)


if __name__ == '__main__':
    main()
