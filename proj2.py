# ECE475 - Frequentist Machine Learning
# Assignment 2
# Jonathan Lam, Tiffany Yu, Harris Paspuleti

# Setting up
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn import preprocessing
import seaborn as sb
from binarylogisticclassifier import BinaryLogisticClassifier
from stepwiselogisticclassifier import StepwiseLogisticClassifier
from l2logisticclassifier import L2LogisticClassifier
from l1logisticclassifier import L1LogisticClassifier
from trinarylogisticclassifier import TrinaryLogisticClassifier

# Import the South African heart disease dataset (in Google Colab)
#!wget https://web.stanford.edu/~hastie/ElemStatLearn/datasets/SAheart.data

# Read through data and create dataset
sahdDataset = pd.read_csv('SAheart.data', index_col=0)

# Textbook drops adiposity and typea
sahdDataset = sahdDataset.drop(['adiposity', 'typea'], axis=1)

# Turn famhist into a quantitative variable
sahdDataset['famhist'] = (sahdDataset['famhist'] == 'Present')*1

# Creates a graph like figure 4.12
# sb.pairplot(dataset, hue = 'chd',palette="hls", height = 3)
# plt.show()

# Generate Features matrix : NxP
sahdDatasetX = sahdDataset.drop(['chd'], axis=1).to_numpy()
# Generate Label matrix : Nx1
sahdDatasety = sahdDataset.loc[:, 'chd'].to_numpy().reshape(-1, 1)

# PART 1: RECREATE TABLE 4.
binaryClassifier = BinaryLogisticClassifier(sahdDatasetX, sahdDatasety)
binaryClassifier.train()
#binaryClassifier.plotIterations()
print(f'theta: {binaryClassifier.theta()}\n% classified correct for unregularized: {np.around((1 - binaryClassifier.pctWrong()) * 100)}%')

# PART 2: STEPWISE
stepwiseClassifier = StepwiseLogisticClassifier(None, None, copySubsetsFrom=binaryClassifier)
stepwiseClassifier.validate()
print(f'theta: {stepwiseClassifier.theta()}\n% classified correct for stepwise: {np.around((1 - stepwiseClassifier.pctWrong()) * 100)}%')

#PART 3: L2 REGULARIZATION
l2Classifier = L2LogisticClassifier(None, None, copySubsetsFrom=binaryClassifier)
l2Classifier.validate()
print(f'theta: {l2Classifier.theta()}\n% classified correct for L2 regularized: {np.around((1 - l2Classifier.pctWrong()) * 100)}%')

# Unregularize converges to slightly higher number than regularized
term = list(sahdDataset.columns.values[:-1])

# STRETCH GOAL 1: L1 REGULARIZATION
l1Classifier = L1LogisticClassifier(None, None, copySubsetsFrom=binaryClassifier)
bestC, coefficients = l1Classifier.validate()
cIterations = np.logspace(-15, 0, 100)
plt.figure()
plt.plot(cIterations, coefficients[:,1:])
plt.xlabel('λ')
plt.ylabel('Coefficients')
plt.title('Lasso Coefficients')

# STRETCH GOAL 2: MULTINOMIAL (> 2 CLASSES) REGRESSION 

# iris dataset for multiclass (3-class regression)
irisDataset = pd.read_csv('iris.data')

# one-hot encode labels
irisDatasety = np.vstack((
    (irisDataset.iloc[:,4] == 'Iris-setosa').to_numpy(dtype=np.float32),
    (irisDataset.iloc[:,4] == 'Iris-versicolor').to_numpy(dtype=np.float32),
    (irisDataset.iloc[:,4] == 'Iris-virginica').to_numpy(dtype=np.float32))).T

# feature matrix
irisDatasetX = irisDataset.iloc[:,:4].to_numpy()

irisClassifier = TrinaryLogisticClassifier(irisDatasetX, irisDatasety)
irisClassifier.trinaryClassificationTrain()
print(f'% wrong on iris dataset: {irisClassifier.pctWrong()}')