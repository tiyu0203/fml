import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn import preprocessing
import seaborn as sb

from binarylogisticclassifier import BinaryLogisticClassifier

# taken mostly literally from (Tsuruoka et al., 2009); involves an
# estimate of the gradient of the L1 norm (abs function) that involves
# some "memory" for improved performance
class L1LogisticClassifier(BinaryLogisticClassifier):

    # apply this after Adam update rule (would be difficult to incorporate with Adam)
    def applyL1Penalty(self):
        for i, theta_i in enumerate(self._theta.reshape(-1)):
            # start from 1 to not penalize the bias
            if i == 0:
                continue

            z = theta_i
            if theta_i > 0:
                self._theta[i,0] = max(0., theta_i - (self._u + self._q[i]))
            elif theta_i < 0:
                self._theta[i,0] = min(0., theta_i + (self._u + self._q[i]))
            self._q[i] += theta_i - z

    def validate(self, iterations=2000):
        # undo l2 regularization
        self._lambda = 0.

        # l1 regularization parameter; C is the letter used in the text
        cIteration = np.logspace(-15, 0, 100)
        bestPctWrong, bestC = float('inf'), None
        pctWrongs = np.zeros_like(cIteration)

        # includes bias for now
        coefficients = np.zeros((cIteration.size, self._theta.size))

        # batch size; for now we use the entire dataset
        self._N = self._subsets['train']['X'].shape[0]
        for j, c in enumerate(cIteration):
            self._C = c
            self._u = 0.
            self._q = np.zeros_like(self._theta).reshape(-1)
            self._theta = np.zeros((self._subsets['train']['X'].shape[1], 1))

            # training loop -- don't use ordinary self.train here
            for i in range(iterations):
                self._u += self._alpha * self._C / self._N
                self.step(i)
                self.applyL1Penalty()
            coefficients[j,:] = self._theta.reshape(-1)
            pctWrong = self.pctWrong(subset='validate')
            pctWrongs[j] = pctWrong
            if pctWrong < bestPctWrong:
                bestPctWrong = pctWrong
                bestC = c

            #print(f'c = {c}, pctWrong: {pctWrong}')

        self._C = bestC

        # retrain with best C
        self._u = 0.
        self._q = np.zeros_like(self._theta).reshape(-1)
        self._theta = np.zeros((self._subsets['train']['X'].shape[1], 1))

        # training loop -- don't use ordinary self.train here
        for i in range(iterations):
            self._u += self._alpha * self._C / self._N
            self.step(i)
            self.applyL1Penalty()

        return self._C, coefficients