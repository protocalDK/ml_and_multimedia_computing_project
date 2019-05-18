import numpy as np
import evaluate
import logistic

trainset = np.load('traindata.npy')
testset = np.load('testdata.npy')


X = trainset[:, :14810]
y = trainset[:, -1]
Xtest = testset[:, :14810]
ytest = testset[:, -1]
