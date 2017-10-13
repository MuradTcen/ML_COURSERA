from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn import datasets
import pandas as pd
import numpy as np


FILENAME = 'housing.data'

def get_X_y(filename):
    X = pd.read_csv(filename, header=None, usecols=list(range(1, 15)))
    y = pd.read_csv(filename, header=None, usecols=[0]).values.reshape(len(X),)
    return X, y

def get_quality_mark(df):
    return KFold(n=len(df), n_folds=5, shuffle=True, random_state=42)

def get_accuracy_classification(X, y, kf, variants):
    k_means = list()
    # for i in range(1, 201):
    for p in variants:
    	# print(i, j)
        kn = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p, metric='minkowski')
        kn.fit(X, y)
        array = cross_val_score(estimator=kn, X=X, y=y,
                                cv=kf, scoring='neg_mean_squared_error')
        m = array.mean()
        k_means.append(m)
    return k_means

def get_max_and_id(lst):
    return max(lst), [i for i, j in enumerate(lst) if max(lst) == j][0]


def main():
	# X, y = get_X_y(FILENAME)
	boston = datasets.load_boston()
	X=pd.DataFrame(data=boston.data,columns=boston.feature_names)
	y=pd.DataFrame(data=boston.target)

	X.dropna(inplace=True)
	X_scale = scale(X)
	kf = get_quality_mark(X)
	variants = np.linspace(1, 10, 200)
	k_means = get_accuracy_classification(X, y, kf, variants)
	print(get_max_and_id(k_means))

if __name__ == '__main__':
	main()
