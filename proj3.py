import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedKFold

# Consider a scenario with N = 50 samples in two equal-sized classes, and p = 5000 quantitative predictors (standard Gaussian) that are independent of the class labels.
#mean = 0, standard deviation = 1
X = np.random.normal(0,1,[50, 5000])
# The true (test) error rate of any classifier is 50%
Y = np.concatenate([np.zeros(25), np.ones(25)])
np.random.shuffle(Y)

#INCORRECT
CV_correct = []
# Screen the predictors: find a subset of “good” predictors that show fairly strong (univariate) correlation with the class labels
X_new = preprocessing.MinMaxScaler().fit_transform(X)
X_new = SelectKBest(chi2, k=100).fit_transform(X_new, Y)
# Using just this subset of predictors, build a multivariate classifier.
#and then using a 1-nearest neighbor classifier,
neigh = KNeighborsClassifier(n_neighbors=1)
# Use cross-validation to estimate the unknown tuning parameters and to estimate the prediction error of the final model.
rkf = RepeatedKFold(n_splits=5, n_repeats=50)

for train_index, test_index in rkf.split(X_new):
    X_train, X_test = X_new[train_index], X_new[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    neigh.fit(X_train, Y_train)
    CV_correct.append(1-neigh.score(X_test, Y_test))
    
print("Average CV Error Rate:", np.array(CV_correct).mean())

X = pd.DataFrame(data=X)
Y = pd.DataFrame(data=Y)

#CORRECT
CV_correct = []

# Use cross-validation to estimate the unknown tuning parameters and to estimate the prediction error of the final model.
rkf = RepeatedKFold(n_splits=5, n_repeats=50)

for train_index, test_index in rkf.split(X):
# Divide the samples into K cross-validation folds (groups) at random.

# For each fold k = 1, 2, . . . , K
# 1. Find a subset of “good” predictors that show fairly strong (univariate) correlation with the class labels, using all of the samples except those in fold k.
# 2. Using just this subset of predictors, build a multivariate classifier, using all of the samples except those in fold k.
# 3. Use the classifier to predict the class labels for the samples in fold k.
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

# Screen the predictors: find a subset of “good” predictors that show fairly strong (univariate) correlation with the class labels
    X_new = preprocessing.MinMaxScaler().fit_transform(X_train)
    X_new = pd.DataFrame(data=X_new)

    kbest = SelectKBest(chi2, k=100)
    kbest.fit_transform(X_new, np.ravel(Y_train))

    best_features = kbest.get_support()
    X_new = X_new.iloc[:, best_features]

    # Using just this subset of predictors, build a multivariate classifier.
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(X_new, np.ravel(Y_train))

    CV_correct.append(1-neigh.score(X_test.iloc[:, best_features], np.ravel(Y_test)))
print("Average CV Error Rate:", np.array(CV_correct).mean())
