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
# import the South African heart disease dataset

#!wget https://web.stanford.edu/~hastie/ElemStatLearn/datasets/SAheart.data
dataset = pd.read_csv('SAheart.data', index_col=0)
# take out the column of adiposity and typea
dataset.pop('adiposity')
dataset.pop('typea')

# We can't have the words present/absent so we replace them with 1 if present and 0 if absent
dataset['famhist'] = (dataset['famhist'] == 'Present')*1

# creates a graph like figure 4.12
#sb.pairplot(dataset, hue = 'chd',palette="hls", height = 3)
# plt.show()



# shuffle and split training into 80% train, 10% validation, 10% test.
# training, validation, test = np.split(dataset.sample(
#     frac=1), [int(.8*len(dataset)), int(.9*len(dataset))])
# prints out the length of each section of the dataset and the dataset itself
# print("Length of dataset:", len(dataset))
# print("Length of training:", len(training))
# print("Length of validation:", len(validation))
# print("Length of test:", len(test))



# seperates the training, testing, and validation into x and y
# famhist is the y, which is the last column
# all but the last columns are the x data
# iloc is Purely integer-location based indexing for selection by position.
# x_train, y_train = training.iloc[:, :-1], training.iloc[:, -1]
# x_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]
# x_val, y_val = validation.iloc[:, :-1], validation.iloc[:, -1]

# normalize all the data
# x_train = preprocessing.normalize(x_train, axis=0)
# x_train = preprocessing.StandardScaler().fit(x_train).transform(x_train)
# x_test = preprocessing.StandardScaler().fit(x_test).transform(x_test)
# x_val = preprocessing.StandardScaler().fit(x_val).transform(x_val)

# add a column of ones for the intercept
# x_train = np.concatenate((np.ones((x_train.shape[0], 1)), x_train), axis=1)
# x_test = np.concatenate((np.ones((x_test.shape[0], 1)), x_test), axis=1)
# x_val = np.concatenate((np.ones((x_val.shape[0], 1)), x_val), axis=1)

# # generate initial weight vector
# N, P = x_train.shape
# P -= 1
# #initialize an array for theta
# theta = np.zeros((P+1, 1))

# theta = np.array([[-4.130, 0.006, 0.080, 0.185, 0.939, -0.035, 0.001, 0.043]]).T

# reshape y_train
#y_train = y_train.to_numpy().reshape((N, 1))


# logistic classifier using SGD w/ Adam optimization
class LogisticClassifier:

    # X: NxP ndarray (features)
    # y: Nx1 ndarray (labels)
    # lam: regularization coefficient
    # alpha: (maximum) learning rate
    def __init__(self, X, y, alpha=0.01):
        N, P = X.shape

        self._alpha = alpha
        self._lambda = 0

        # add column of 1's to X
        X = np.hstack((np.ones((N, 1)), X))

        # seperates the training, testing, and validation into x and y
        # famhist is the y, which is the last column
        # all but the last columns are the x data
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
        print("Length of dataset:", len(dataset))
        print("Length of training:", split1)
        print("Length of validation:", abs(split1-split2))
        print("Length of test:", len(dataset) - split2)

        # initialize weight vector
        # TODO: separate the bias from
        self._theta = np.zeros((P+1, 1))

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
        return y.T @ np.log(self.h(X)) + (1 - y).T @ np.log(1 - self.h(X)) - (self._lambda * (self._theta[1:,:] ** 2))

    # percent classified wrong on training subset
    def pctWrong(self, subset='test'):
        X, y = self._subsets[subset]['X'], self._subsets[subset]['y']
        N, _ = X.shape
        return np.sum(np.abs(np.round(self.h(X)) - y)) / N

    def step(self, includeMask=None):
        # update adam moments
        thetaGrad = self.grad(self._subsets['train']['X'], self._subsets['train']['y'])
        # weighted average of the gradient
        self._ztheta = self._beta1 * self._ztheta + (1 - self._beta1) * thetaGrad
        # weighted average of the square gradient
        self._zthetaSquared = self._beta2 * self._zthetaSquared + (1 - self._beta2) * thetaGrad ** 2

        # adam update rule
        self._theta += self._alpha * self._ztheta / (np.sqrt(self._zthetaSquared) + self._ep)

        # exclude certain features
        if includeMask is not None:
            self._theta *= includeMask

    def train(self, iterations=2000, includeMask=None):
        self._theta = np.zeros((self._subsets['train']['X'].shape[1], 1))
        # trainLikelihood = np.zeros((iterations, 1))
        # validateLikelihood = np.zeros((iterations,1))
        for i in range(iterations):
            self.step(includeMask)
        #     trainLikelihood[i] = np.mean(self.l(subset = 'train'))
        #     validateLikelihood[i] = np.mean(self.l(subset = 'validate'))
        # if l2 == 0:
        #   plt.plot(np.linspace(0, iterations, iterations),trainLikelihood)
        #   plt.plot(np.linspace(0,iterations, iterations),validateLikelihood)
        #   plt.legend(['Training loglikelihood', 'Validate loglikelihood'])
        #   plt.show()
        # else:
        #   pass
            # if i % 1000 == 0:
            #     print(f'iteration {i}\tclassified wrong: {np.around(self.pctWrong(),2)}\tlog likelihood: {np.around(self.l(subset="train"),2)}')

    def plotIterations(self, iterations=2000):
        for i in range(iterations):
            trainLikelihood[i] = np.mean(self.l(subset = 'train'))
            validateLikelihood[i] = np.mean(self.l(subset = 'validate'))

        plt.plot(np.linspace(0, iterations, iterations),trainLikelihood)
        plt.plot(np.linspace(0,iterations, iterations),validateLikelihood)
        plt.legend(['Training loglikelihood', 'Validate loglikelihood'])
        plt.show() 

    def theta(self):
        return self._theta

    def stepWise(self):
        _, P = self._subsets['train']['X'].shape
        P -= 1

        # list of features to exclude and include
        exclude = list(range(P))
        include = []
        # list of features to include
        includeMask = np.zeros((P+1, 1)) 
        includeMask[0] = 1

        pctWrongs = np.zeros((P+1, 1))
        #calculate the percent wrong relative to the validate set of data
        pctWrongs[0] = 1 - np.mean(self._subsets['validate']['y'])

        # loops over number of features in model
        for i in range(P):

            # find best next feature to include
            bestPctWrong, bestFeature = float('inf'), None
            for feature in exclude:
                # copy includeMask into currentIncludeMask, unmask feature
                currentIncludeMask = np.array(includeMask)
                currentIncludeMask[feature+1] = 1

                # train on currentIncludeMask
                self.train(includeMask=currentIncludeMask)

                # calculate percent wrong on validation set
                #Trying to find out when it gives you the least error
                pctWrong = self.pctWrong(subset='validate')
                if pctWrong < bestPctWrong:
                    bestPctWrong = pctWrong
                    bestFeature = feature

            # minimize percent wrong
            pctWrongs[i+1] = bestPctWrong

            # add feature to includeMask, remove from exclude
            exclude.remove(bestFeature)
            include.append(bestFeature)
            includeMask[bestFeature] = 1

        # find minimum of pctWrongs
        bestNumFeatures = np.argwhere(pctWrongs == np.min(pctWrongs))[0,0]
        bestIncludeMask = np.zeros((P+1, 1))
        bestIncludeMask[0] = 1
        for i in range(bestNumFeatures):
            bestIncludeMask[include[i]+1] = 1

        # retrain with best include mask, return theta
        self.train(includeMask=bestIncludeMask)
        return self._theta, include[:bestNumFeatures]

    def l2Regularize(self):
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

    # taken very literally from (Tsuruoka et al., 2009)
    # apply this after Adam update rule (would be difficult to incorporate with Adam)
    def applyL1Penalty(self):
        for i, theta_i in enumerate(self._theta.reshape(-1)):
            z = theta_i
            if theta_i > 0:
                self._theta[i,0] = max(0., theta_i - (self._u + self._q[i]))
            elif theta_i < 0:
                self._theta[i,0] = min(0., theta_i + (self._u + self._q[i]))
            self._q[i] += theta_i - z

    # taken very literally from (Tsuruoka et al., 2009)
    def l1RegularizationTrain(self, iterations=2000, includeMask=None):
        self._u = 0.
        self._q = np.zeros_like(self._theta).reshape(-1)
        self._theta = np.zeros((self._subsets['train']['X'].shape[1], 1))

        # l1 regularization parameter; C is the letter used in the text
        self._C = 0.1

        # batch size; for now we use the entire dataset
        self._N = self._subsets['train']['X'].shape[0]

        for i in range(iterations):
            self._u += self._alpha * self._C / self._N
            self.step(includeMask)
            self.applyL1Penalty()

# PART 1: RECREATE TABLE 4.2
X = dataset.drop(['chd'], axis=1).to_numpy()
y = dataset.loc[:, 'chd'].to_numpy().reshape(-1, 1)
classifier = LogisticClassifier(X, y)
classifier.train()
print(f'theta: {classifier.theta()}\n% classified correct for unregularized: {np.around((1 - classifier.pctWrong()) * 100)}%')

# PART 2: STEPWISE
# classifier.stepWise()
# print(f'theta: {classifier.theta()}\n% classified correct for stepwise: {np.around((1 - classifier.pctWrong()) * 100)}%')

#PART 3: L2 REGULARIZATION
# classifier.l2Regularize()
# print(f'theta: {classifier.theta()}\n% classified correct for L2 regularized: {np.around((1 -classifier.pctWrong()) * 100)}%')

#unregularize converge to slightly higher number than regularized
classifier.l1RegularizationTrain()
print(f'theta: {classifier.theta()}\n% classified correct for L1 regularized: {np.around((1 - classifier.pctWrong()) * 100)}%')