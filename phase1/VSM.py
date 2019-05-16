import os
import sys
from phase1 import general
from phase1 import tokenizing


data_path = general.check_data_path("Sarcasm_Headlines_Dataset.json")
dataset = open(data_path)
data = [eval(i) for i in dataset]

preprocess_result_path = general.create_result_preprocessing()


headlines = []
for i in data:
    headlines.append(i["headline"])

labels = []
for i in data:
    labels.append(i["is_sarcastic"])


vocab =set()
for headline in headlines:
    terms = tokenizing.get_terms(headline)
    vocab.update(terms)
    print(len(vocab))
    #frequency_of_words = Counter(terms)


general.write_file(os.path.join(preprocess_result_path, "dictionary1.txt"), vocab)
