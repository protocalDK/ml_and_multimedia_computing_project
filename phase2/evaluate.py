from sklearn import metrics

"""
    calculate evaluation score for model
"""


def score(y, y_pred):
    accuracy = metrics.accuracy_score(y, y_pred)
    pre = metrics.precision_score(y, y_pred)
    re = metrics.recall_score(y, y_pred)
    f1 = metrics.f1_score(y, y_pred)

    return accuracy, pre, re, f1


def output(model_name, y, y_pred):
    accuracy, pre, re, f1 = score(y, y_pred)

    print('"' + model_name + '"')
    print("Accuracy  : ", accuracy)
    print("Precision : ", pre)
    print("Recall    : ", re)
    print("F1 Score  : ", f1)
