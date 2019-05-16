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
preprocess_result_path = general.create_result_preprocessing()
""" Đọc data trong file"""
dataset = open(data_path)
text_data = [eval(i) for i in dataset]


headlines = list()
""" List chứa mỗi dòng headline trong 1 phần tử"""
for i in text_data:
    headlines.append(i["headline"])


labels = list()
""" List chứa label tương ứng của headlines"""
for i in text_data:
    labels.append(i["is_sarcastic"])


#""" Tạo một tập từ điển chứa tất các term xuất hiện trong data.
#    Lưu các term vào một set để tránh việc lưu trùng.
#    Sau khi có được set các term, tiến hành lưu vào file tên "dictionary1.txt"."""
#vocab =set()
#for headline in headlines:
#    terms = tokenizing.get_terms(headline)
#    vocab.update(terms)
#    print(len(vocab))
#
#general.write_file(os.path.join(preprocess_result_path, "dictionary1.txt"), vocab)
#
#
#""" Lấy các term từ từ điển được tạo trước đó"""
#vocab = general.file_to_set(os.path.join(preprocess_result_path, "dictionary1.txt"))

print("Getting files ...")
corpus = list()
for headline in headlines:
    terms = tokenizing.get_terms(headline)
    bag_of_words = Counter(terms)
    corpus.append(bag_of_words)
    #print(len(corpus))

print("Building vector space model...")
data = helpers.compute_weight(corpus)


np.save("data.npy", data)
#array_reloaded = np.load('fnumpy.npy')
