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

""" Đọc data trong file"""
dataset = open(data_path)
data = [eval(i) for i in dataset]


""" Xử lý dữ liệu cho trainset"""
text_train_data = data[:20000]

headlines = list()
""" List chứa mỗi dòng headline trong 1 phần tử"""
labels = list()
""" List chứa label tương ứng của headlines"""
for i in text_train_data:
    headlines.append(i["headline"])
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
tf_train = helpers.compute_tf(corpus)

print("Building vector space model...")
traindata = helpers.compute_weight(tf_train, idf)
print("...Done")

for i in range(0, len(traindata)):
    traindata[i].append(labels[i])


np.save("traindata.npy", traindata)
#array_reloaded = np.load('traindata.npy')


""" Xử lý dữ liệu cho testset"""
text_test_data = data[20000:]

headlines_ = list()
""" List chứa mỗi dòng headline trong 1 phần tử"""
labels_ = list()
""" List chứa label tương ứng của headlines"""

for i in text_test_data:
    headlines_.append(i["headline"])
    labels_.append(i["is_sarcastic"])


print("Getting files ...")
corpus = list()
for headline in headlines_:
    terms = tokenizing.get_terms(headline)
    bag_of_words = Counter(terms)
    corpus.append(bag_of_words)
    #print(len(corpus))
print("...Done")


tf_test = helpers.compute_tf(corpus)

print("Building vector space model...")
testdata = helpers.compute_weight(tf_test, idf)
print("...Done")

for i in range(0, len(testdata)):
    testdata[i].append(labels_[i])


#np.save("testdata.npy", testdata)
