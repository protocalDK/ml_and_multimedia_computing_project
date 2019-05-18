from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm

"""
    K-Nearest Neighbors using sklearn
"""


def KNN(X, Xtest, y):
    model = KNeighborsClassifier(
        n_neighbors=7, metric='euclidean').fit(X, y)
    p = model.predict(Xtest)

    return p


"""
    Decision Tree using sklearn
"""


def DecisionTree(X, Xtest, y):
    model = DecisionTreeClassifier().fit(X, y)
    p = model.predict(Xtest)

    return p


"""
    Naive Bayes using sklearn
"""


def NaiveBayes(X, Xtest, y):
    model = BernoulliNB().fit(X, y)
    p = model.predict(Xtest)

    return p


"""
    Support Vector Machine using sklearn
"""


def SupportVectorMachine(X, Xtest, y):
    model = svm.SVC(kernel='linear')
    model.fit(X, y)
    p = model.predict(Xtest)

    return p
