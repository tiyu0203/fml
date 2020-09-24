import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn import preprocessing
import seaborn as sb

from baselogisticclassifier import BaseLogisticClassifier

class BinaryLogisticClassifier(BaseLogisticClassifier):

    # hypothesis function; uses trained theta
    # y_hat
    # returns N x 1
    def h(self, X):
        return 1 / (1 + np.exp(-X @ self._theta))

    # update function
    # theta_j := theta_j + alpha(y_i -h_theta(x_i)) * x_i_j
    # returns (P+1)
    # SGD =  j+α(y(i)−hθ(x(i)))x(i)j
    def grad(self, X, y):
        # don't penalize the bias
        return X.T @ (y - self.h(X)) - 2 * self._lambda * np.vstack((np.zeros((1,1)), self._theta[1:,:]))

    # WE DON'T USE THIS FUNCTION ANYWHERE
    # log likelihood
    def l(self, subset):
        X, y = self._subsets[subset]['X'], self._subsets[subset]['y']
        #print('TESTING SHAPE', y.T @ np.log(self.h(X)) + (1 - y).T @ np.log(1 - self.h(X)) - (self._lambda * (self._theta[1:,:] ** 2)).shape)
        return y.T @ np.log(self.h(X)) + (1 - y).T @ np.log(1 - self.h(X)) #- (self._lambda * (self._theta[1:,:] ** 2))

    # percent classified wrong on training subset
    def pctWrong(self, subset='test'):
        X, y = self._subsets[subset]['X'], self._subsets[subset]['y']
        N, _ = X.shape
        return np.sum(np.round(np.abs(self.h(X) - y))) / N

    def step(self, iter, includeMask=None):
        # update adam moments
        thetaGrad = self.grad(self._subsets['train']['X'], self._subsets['train']['y'])
        # weighted average of the gradient
        self._ztheta = self._beta1 * self._ztheta + (1 - self._beta1) * thetaGrad
        # weighted average of the square gradient
        self._zthetaSquared = self._beta2 * self._zthetaSquared + (1 - self._beta2) * thetaGrad ** 2

        # bias-corrected moments
        bcZTheta = self._ztheta / (1 - self._beta1 ** (iter + 1))
        bcZThetaSquared = self._zthetaSquared / (1 - self._beta2 ** (iter + 1))

        # adam update rule
        self._theta += self._alpha * bcZTheta / (np.sqrt(bcZThetaSquared) + self._ep)

        # exclude certain features
        if includeMask is not None:
            self._theta *= includeMask

    def train(self, iterations=2000, includeMask=None):
        self._loglikelihoods = np.zeros(iterations)

        self._theta = np.zeros((self._subsets['train']['X'].shape[1], 1))
        # trainLikelihood = np.zeros((iterations, 1))
        # validateLikelihood = np.zeros((iterations,1))
        for i in range(iterations):
            self.step(i, includeMask)

            self._loglikelihoods[i] = self.l(subset='train')

            # if i % 1000 == 0:
            #     print(f'iteration {i}\tclassified wrong: {np.around(self.pctWrong(),2)}\tlog likelihood: {np.around(self.l(subset="train"),2)}')