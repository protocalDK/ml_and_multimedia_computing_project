import numpy as np
import copy
from sklearn.linear_model import LogisticRegression


"""
    SIGMOID comput sigmoid function
"""


def sigmoid(z):
    #g = np.zeros((len(z)),1)
    g = 1.0/(1 + np.exp(-z))

    return g


"""
    COSTFUNCTION compute cost for logistic regression
"""


def costFunction(theta, X, y):

    """ number of training examples"""

    m = len(y)

    """ initial return values"""

    J = 0

    """ compute cost"""

    h = sigmoid(X.dot(theta))
    J = 1.0/m * ((-y).transpose().dot(np.log(h)) -
                 (1 - y).transpose().dot(np.log(1 - h)))

    return J


"""
    BGD compute batch gradient for logistic regression to learn theta
    theta updates by taking num_iters gradient steps woth learning rate alpha
"""


def BGD(X, y, theta, alpha, num_iters=1000):

    """initial some values"""

    m = len(y)
    J_history = np.zeros((num_iters, 1))

    for iter in range(0, num_iters):

        """ perform a single gradient step on the parameter vector theta."""

        h = sigmoid(X.dot(theta))
        theta = theta - (alpha / m) * X.transpose().dot((h - y))

        """ save the cost J in every iteration"""
        
        J_history[iter] = costFunction(theta, X, y)
        #print(iter)

    return theta, J_history


"""
  PREDICT predict whether the label is 0 or 1 using learned regression parameters theta
"""


def predict_handwrite(theta, X, threshold=0.5):
    """
        p = PREDICT(theta, X) computes the predictions for X using a threshold at 0.5 (default)
        (i.e., if sigmoid(theta'*x) >= threshold, predict 1)
    """

    """ number of training examples"""

    m = X.shape[0]

    """ return values"""

    p = np.zeros((m, 1))

    for i in range(0, m):
        print(sigmoid(X[i, :].dot(theta)))
        if sigmoid(X[i, :].dot(theta)) >= threshold:
            p[i] = 1
        else:
            p[i] = 0

    return p


def LogisticRegression_handwrite(X_, Xtest_, y_, alpha, num_iters=1000, threshold=0.5):
    X = copy.deepcopy(X_)
    y = copy.deepcopy(y_)
    Xtest = copy.deepcopy(Xtest_)

    y = y.reshape((len(y), 1))

    """
        setup the data matrix appropriately
        and add ones for the intercept term
    """

    m, n = X.shape
    mtest = Xtest.shape[0]

    """ add intercept term to X"""

    X = np.hstack((np.ones((m, 1)), X))
    Xtest = np.hstack((np.ones((mtest, 1)), Xtest))

    """ initialize fitting parameters"""

    initial_theta = np.zeros((n + 1, 1))

    """ run BGD to obtain the optimal theta"""

    theta, cost = BGD(X, y, initial_theta, alpha=alpha, num_iters=num_iters)

    """ get prediction on testset"""

    p = predict_handwrite(theta, Xtest, threshold=threshold)

    return p, cost



"""
    logistic Regression using sklearn 
"""

def LogisticRegression_sklearn(X, Xtest, y):
    model = LogisticRegression(solver='saga')
    model.fit(X, y)
    p = model.predict(Xtest)

    return p
