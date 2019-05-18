import os
import sys
from phase1 import general
from phase1 import tokenizing
from phase1 import helpers
#import general
#import tokenizing
#import helpers
from collections import Counter
import numpy as np


data_path = general.check_data_path("Sarcasm_Headlines_Dataset.json")

""" read file"""

dataset = open(data_path)
data = [eval(i) for i in dataset]


"""
    preprocess data on trainset
"""

text_train_data = data[:20000]

headlines = list()

""" get each headline on trainset"""

labels = list()

""" get label on trainset"""

for i in text_train_data:
    headlines.append(i["headline"])
    labels.append(i["is_sarcastic"])


"""
    make corpus of training headlines
"""

print("Getting files ...")
corpus = list()
for headline in headlines:
    terms = tokenizing.get_terms(headline)

    """
        Counter help counting frequency of each term in doc
        and return a dict
    """

    bag_of_words = Counter(terms)
    corpus.append(bag_of_words)
    #print(len(corpus))
print("...Done")


"""
    compute tf-idf matrix on trainset
"""

idf = helpers.compute_idf(corpus)
tf_train = helpers.compute_tf(corpus)

print("Building vector space model...")
traindata = helpers.compute_weight(tf_train, idf)
print("...Done")


"""
    add label into it
"""

for i in range(0, len(traindata)):
    traindata[i].append(labels[i])


np.save("traindata.npy", traindata)
#array_reloaded = np.load('traindata.npy')


"""
    preprocess data on testset
"""

text_test_data = data[20000:]

headlines_ = list()

""" get each headline on testset"""

labels_ = list()

""" get label on testset"""

for i in text_test_data:
    headlines_.append(i["headline"])
    labels_.append(i["is_sarcastic"])


"""
    make corpus of training headlines
"""

print("Getting files ...")
corpus = list()
for headline in headlines_:
    terms = tokenizing.get_terms(headline)
    bag_of_words = Counter(terms)
    corpus.append(bag_of_words)
    #print(len(corpus))
print("...Done")


"""
    compute tf-idf matrix on testset
    using trainset's idf vector
"""

tf_test = helpers.compute_tf(corpus)

print("Building vector space model...")
testdata = helpers.compute_weight(tf_test, idf)
print("...Done")


"""
    add label into it
"""

for i in range(0, len(testdata)):
    testdata[i].append(labels_[i])


#np.save("testdata.npy", testdata)
