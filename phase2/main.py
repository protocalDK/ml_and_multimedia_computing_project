import numpy as np
import evaluate
import logistic
import other_models


trainset = np.load('traindata.npy')
testset = np.load('testdata.npy')


X = trainset[:, :-1]
y = trainset[:, -1]
Xtest = testset[:, :-1]
ytest = testset[:, -1]


"""
    Logictis Regression handwrite
"""

alpha = 5
num_iters = 1000
threshold = 0.5

p, cost = logistic.LogisticRegression_handwrite(
    X, Xtest, y, alpha, num_iters, threshold)

print(evaluate.output("Logictis Regression handwrite", ytest, p))


"""
    Logistic Regression using sklearn
"""

p_l = logistic.LogisticRegression_sklearn(X, Xtest, y)
print(evaluate.output("Logictis Regression sklearn", ytest, p_l))


"""
    K-Nearest Neighbors using sklearn
"""

p_k = other_models.KNN(X, Xtest, y)
print(evaluate.output("K-Nearest Neighbors sklearn", ytest, p_k))



"""
    Decision Tree using sklearn
"""

p_d = other_models.DecisionTree(X, Xtest, y)
print(evaluate.output("Decision Tree sklearn", ytest, p_d))


"""
    Naive Bayes using sklearn
"""

p_n = other_models.NaiveBayes(X, Xtest, y)
print(evaluate.output("Naive Bayes sklearn", ytest, p_n))


"""
    Support Vector Machine using sklearn
"""

p_s = other_models.SupportVectorMachine(X, Xtest, y)
print(evaluate.output("Support Vector Machine sklearn", ytest, p_s))
