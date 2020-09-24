import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn import preprocessing
import seaborn as sb

from binarylogisticclassifier import BinaryLogisticClassifier

class L2LogisticClassifier(BinaryLogisticClassifier):

    def validate(self):
        #create a bunch of lambdas in order to iterate through them
        lams = np.logspace(-20, 5, 100)
        
        # standardize
        # whenever you regularize when weights are multplied times features you should normalize
        self._subsets['train']['X'] = preprocessing.StandardScaler().fit(self._subsets['train']['X']).transform(self._subsets['train']['X'])
        self._subsets['validate']['X'] = preprocessing.StandardScaler().fit(self._subsets['validate']['X']).transform(self._subsets['validate']['X'])
        self._subsets['test']['X'] = preprocessing.StandardScaler().fit(self._subsets['test']['X']).transform(self._subsets['test']['X'])

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