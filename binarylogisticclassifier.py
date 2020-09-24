import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn import preprocessing
import seaborn as sb

from baselogisticclassifier import LogisticClassifier

class BinaryLogisticClassifier(LogisticClassifier):

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