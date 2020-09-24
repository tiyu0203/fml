import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn import preprocessing
import seaborn as sb

# base logistic classifier class; includes some common utilities;
# this class is abstract and doesn't contain a train() or validate() method,
# which must be implemented in its inheritors
class BaseLogisticClassifier:

    # X: NxP ndarray (features)
    # y: Nx1 ndarray (labels)
    # alpha: (maximum) learning rate
    # copySubsetsFrom: treat this as a copy constructor, copy
    # over the subsets from a different BaseLogisticClassifier
    def __init__(self, X, y, alpha=0.01, copySubsetsFrom=None):
        self._alpha = alpha
        self._lambda = 0

        # copySubsetsFrom is provided; copy subsplits
        if copySubsetsFrom is not None:
            self._subsets = copySubsetsFrom._subsets
            P = self._subsets['train']['X'].shape[1] - 1

        # X, y are provided; manually split subsets
        else:
            N, P = X.shape

            # add column of 1's to X
            X = np.hstack((np.ones((N, 1)), X))

            # randomly split data into training, validation, test
            indices = np.arange(N)
            np.random.shuffle(indices)
            split1, split2 = int(N*0.8), int(N*0.9)
            self._subsets = {
                'train': {
                    'X': X[indices[:split1], :],
                    'y': y[indices[:split1], :]
                },
                'validate': {
                    'X': X[indices[split1:split2], :],
                    'y': y[indices[split1:split2], :]
                },
                'test': {
                    'X': X[indices[split2:], :],
                    'y': y[indices[split2:], :]
                }
            }

            # print the lengths of the dataset and each set
            print("Length of dataset:", N)
            print("Length of training:", split1)
            print("Length of validation:", split2-split1)
            print("Length of test:", N-split2)

        # initialize weight vector (includes the bias, hence the P+1)
        self._theta = np.zeros((P+1, 1))

        # intitialize Adam coefficients
        # Adam is an optimization algorithm that can be 
        #used instead of the classical stochastic gradient descent procedure 
        #to update network weights iterative based in training data.
        # adam optimizer
        # beta1. The exponential decay rate for the first moment estimates (e.g. 0.9).
        self._beta1 = 0.9
        # beta2. The exponential decay rate for the second-moment estimates (e.g. 0.999). 
        # This value should be set close to 1.0 on problems with a sparse gradient (e.g. NLP and computer vision problems).
        self._beta2 = 0.999
        # Return an array of zeros with the same shape and type as a given array.
        self._ztheta = np.zeros_like(self._theta)
        self._zthetaSquared = np.zeros_like(self._theta)
        # epsilon. Is a very small number to prevent any division by zero in the implementation (e.g. 10E-8).
        self._ep = 0.0001

    def getLogLikelihoods(self):
        return self._loglikelihoods

    def plotLoglikelihood(self):
        iterations = range(0,2000)
        plt.figure()
        plt.plot(iterations, self._loglikelihoods)
        plt.xlabel('Iterations')
        plt.ylabel('Loglikelihoods') 

    def theta(self):
        return self._theta