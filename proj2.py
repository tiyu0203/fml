# ECE475
# Assignment 2
# Jonathan Lam, Tiffany Yu, Harris Paspuleti

# Setting up
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn import preprocessing
import seaborn as sb
#import the South African heart disease dataset

#!wget https://web.stanford.edu/~hastie/ElemStatLearn/datasets/SAheart.data
dataset = pd.read_csv('SAheart.data', index_col=0)
#take out the column of adiposity and typea
dataset.pop('adiposity')
dataset.pop('typea')
#add a column of ones for the intercept
# dataset.insert(0, 'intercept', 1)
#We can't have the words present/absent so we replace them with 1 if present and 0 if absent
dataset['famhist'] = (dataset['famhist'] == 'Present')*1

#shuffle and split training into 80% train, 10% validation, 10% test.
training, validation, test = np.split(dataset.sample(frac=1), [int(.8*len(dataset)), int(.9*len(dataset))])
#prints out the length of each section of the dataset and the dataset itself
print("Length of dataset:", len(dataset))
print( "Length of training:", len(training))
print("Length of validation:", len(validation))
print("Length of test:", len(test))

#creates a graph like figure 4.12
#sb.pairplot(dataset, hue = 'chd',palette="hls", height = 3)
# plt.show()

#seperates the training, testing, and validation into x and y
#famhist is the y, which is the last column
#all but the last columns are the x data
#iloc is Purely integer-location based indexing for selection by position.
x_train, y_train = training.iloc[:, :-1], training.iloc[:, -1]
x_test, y_test = test.iloc[:, :-1], test.iloc[:,-1]
x_val, y_val = validation.iloc[:, :-1], validation.iloc[:,-1]

#normalize all the data
# x_train = preprocessing.normalize(x_train, axis=0)
# x_train = preprocessing.StandardScaler().fit(x_train).transform(x_train)
# x_test = preprocessing.StandardScaler().fit(x_test).transform(x_test)
# x_val = preprocessing.StandardScaler().fit(x_val).transform(x_val)
#add a column of ones for the intercept
x_train = np.concatenate((np.ones((x_train.shape[0], 1)),x_train), axis = 1)
x_test = np.concatenate((np.ones((x_test.shape[0], 1)),x_test), axis = 1)
x_val = np.concatenate((np.ones((x_val.shape[0], 1)),x_val), axis = 1)

# generate initial weight vector
N, P = x_train.shape
P -= 1
theta = np.zeros((P+1, 1))

# theta = np.array([[-4.130, 0.006, 0.080, 0.185, 0.939, -0.035, 0.001, 0.043]]).T

# reshape y_train
y_train = y_train.to_numpy().reshape((N, 1))


# logistic classifier using SGD w/ Adam optimization
class LogisticClassifier:

    # X: NxP ndarray (features)
    # y: Nx1 ndarray (labels)
    # lam: regularization coefficient
    # alpha: (maximum) learning rate
    def __init__(self, X, y, lam=0., alpha=0.001):
        N, P = X.shape

        self._lam = lam
        self._alpha = alpha

        # add column of 1's to X
        X = np.hstack((np.ones((N, 1)), X))

        # randomly split data into training, validation, test
        indices = np.arange(N)
        np.random.shuffle(indices)
        split1, split2 = int(N*0.8), int(N*0.9)
        self._subsets = {
            'train': {
                'X': X[indices[:split1],:],
                'y': y[indices[:split1],:]
            },
            'validate': {
                'X': X[indices[split1:split2],:],
                'y': y[indices[split1:split2],:]
            },
            'test': {
                'X': X[indices[split2:],:],
                'y': y[indices[split2:],:]
            }
        }

        # generate initial weight vector
        # TODO: separate the bias from 
        self._theta = np.zeros((P+1, 1))

        # adam optimizer
        self._beta1 = 0.9
        self._beta2 = 0.999
        self._ztheta = np.zeros_like(self._theta)
        self._zthetaSquared = np.zeros_like(self._theta)
        self._ep = 0.0001

    # hypothesis function; uses trained theta
    #y_hat
    # returns N x 1
    def h(self, X):
        return 1 / (1 + np.exp(-X @ self._theta))

    # update function
    # theta_j := theta_j + alpha(y_i -h_theta(x_i)) * x_i_j
    # returns (P+1)
    #SGD =  j+α(y(i)−hθ(x(i)))x(i)j
    def grad(self, X, y):
        # return gradient of l(theta, X, y) w.r.t. theta
        return X.T @ (y - self.h(X)) - 2 * self._lam * np.linalg.norm(self._theta)

    # log likelihood
    def l(self, X, y):
        return y.T @ np.log(self.h(X)) + (1 - y).T @ np.log(1 - self.h(X)) - (self._lam * np.linalg.norm(self._theta ** 2))

    # percent classified wrong on training subset
    def pctWrong(self, subset='test'):
        X, y = self._subsets[subset]['X'], self._subsets[subset]['y']
        N, _ = X.shape
        return np.sum(abs(np.round(self.h(X)) - y)) / N

    def step(self, includeMask=None):
        # update adam moments
        thetaGrad = self.grad(self._subsets['train']['X'], self._subsets['train']['y'])
        #weighted average of the gradient
        self._ztheta = self._beta1 * self._ztheta + (1 - self._beta1) * thetaGrad
        #weighted average of the square gradient
        self._zthetaSquared = self._beta2 * self._zthetaSquared + (1 - self._beta2) * thetaGrad ** 2

        # adam update rule
        self._theta += self._alpha * self._ztheta / (np.sqrt(self._zthetaSquared) + self._ep)

        # exclude certain features
        if includeMask is not None:
            self._theta *= includeMask

    def train(self, iterations=10000, includeMask=None):
        for i in range(iterations):
            self.step(includeMask)

            if i % 1000 == 0:
                print(f'iteration {i}\tclassified wrong: {np.around(self.pctWrong(),2)}\tlog likelihood: {np.around(self.l(x_train, y_train),2)}')

    def theta(self):
        return self._theta

    def stepWise(self):
        _, P = self._subsets['train']['X'].shape
        P -= 1

        # list of features to exclude
        exclude = list(range(P))
        # list of features to include
        includeMask = np.zeros((P+1, 1))
        includeMask[0] = 1

        for feature in exclude:
            # copy includeMask into currentIncludeMask, unmask feature
            currentIncludeMask = np.array(includeMask)
            currentIncludeMask[feature+1] = 1

            # train on currentIncludeMask
            self._theta = np.zeros((P+1, 1))
            self.train(includeMask=currentIncludeMask)
            print(self._theta)

### PART 1: RECREATE TABLE 4.2
X = dataset.drop(['chd'], axis=1).to_numpy()
y = dataset.loc[:,'chd'].to_numpy().reshape(-1, 1)
classifier = LogisticClassifier(X, y)
# classifier.train()
# print(f'theta: {classifier.theta()}\n% classified wrong: {np.around(classifier.pctWrong() * 100)}%')

### PART 2: STEPWISE
classifier.stepWise()