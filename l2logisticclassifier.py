import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn import preprocessing
import seaborn as sb

from binarylogisticclassifier import BinaryLogisticClassifier

class L2LogisticClassifier(BinaryLogisticClassifier):

    # make sure to standardize features
    def __init__(self, X, Y, alpha=0.01, copySubsetsFrom=None):
        super().__init__(X, Y, alpha=alpha,
                         copySubsetsFrom=copySubsetsFrom,
                         standardizeFeatures=True)

    # update function with L2 penalty
    # theta_j := theta_j + alpha(y_i -h_theta(x_i)) * x_i_j
    # returns (P+1)
    # SGD =  j+α(y(i)−hθ(x(i)))x(i)j
    def grad(self, X, y):
        # don't penalize the bias
        return X.T @ (y - self.h(X)) - 2 * self._lambda * np.vstack((np.zeros((1,1)), self._theta[1:,:]))

    def validate(self):
        #create a bunch of lambdas in order to iterate through them
        lams = np.logspace(-20, 5, 100)
        
        # FIXME: this is terrible
        #Removing the ones because we don't want to regularize the bias term
        P = self._subsets['train']['X'].shape[1] - 1
        self._subsets['train']['X'][0,:] = np.ones((1, P+1))
        self._subsets['validate']['X'][0,:] = np.ones((1, P+1))
        self._subsets['test']['X'][0,:] = np.ones((1, P+1))

        bestPctWrong, bestLambda = float('inf'), None
        pctWrongs = np.zeros_like(lams)

        for i, lam in enumerate(lams):
            self._lambda = lam
            # print(lam, self._lambda)
            self.train()

            pctWrong = self.pctWrong(subset='validate')
            pctWrongs[i] = pctWrong
            # calculate percent wrong on validation set
            #Trying to find out when it gives you the least error
            if pctWrong < bestPctWrong:
                bestPctWrong = pctWrong
                bestLambda = lam

            #print(f'lam: {lam}\tpctWrong: {pctWrong}')

        #print(bestPctWrong, bestLambda, pctWrongs)
        self._lambda = bestLambda
        self.train()
        return self._theta