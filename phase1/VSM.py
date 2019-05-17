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


traindata_path = general.check_data_path("Sarcasm_Headlines_Trainset.json")
testdata_path = general.check_data_path("Sarcasm_Headlines_Testset.json")

""" Xử lý dữ liệu cho trainset"""

""" Đọc data trong file"""
trainset = open(traindata_path)
text_train_data = [eval(i) for i in trainset]

headlines = list()
""" List chứa mỗi dòng headline trong 1 phần tử"""
for i in text_train_data:
    headlines.append(i["headline"])

labels = list()
""" List chứa label tương ứng của headlines"""
for i in text_train_data:
    labels.append(i["is_sarcastic"])

print("Getting files ...")
corpus = list()
for headline in headlines:
    terms = tokenizing.get_terms(headline)
    bag_of_words = Counter(terms)
    corpus.append(bag_of_words)
    #print(len(corpus))
print("...Done")

idf = helpers.compute_idf(corpus)
tf = helpers.compute_tf(corpus)

print("Building vector space model...")
traindata = helpers.compute_weight(tf, idf)
print("...Done")

for i in range(0, len(traindata)):
    traindata[i].append(labels[i])


np.save("traindata.npy", traindata)
#array_reloaded = np.load('traindata.npy')


""" Xử lý dữ liệu cho testset"""

""" Đọc data trong file"""
testset = open(testdata_path)
text_test_data = [eval(i) for i in testset]

headlines_ = list()
""" List chứa mỗi dòng headline trong 1 phần tử"""
for i in text_test_data:
    headlines.append(i["headline"])

labels_ = list()
""" List chứa label tương ứng của headlines"""
for i in text_test_data:
    labels.append(i["is_sarcastic"])

print("Getting files ...")
corpus = list()
for headline in headlines_:
    terms = tokenizing.get_terms(headline)
    bag_of_words = Counter(terms)
    corpus.append(bag_of_words)
    #print(len(corpus))
print("...Done")

tf = helpers.compute_tf(corpus)

print("Building vector space model...")
testdata = helpers.compute_weight(tf, idf)
print("...Done")

for i in range(0, len(testdata)):
    testdata[i].append(labels_[i])


np.save("testdata.npy", testdata)
