import os
import sys
from general import *
from preprocessing import *


data_path = check_data_path("Sarcasm_Headlines_Dataset.json")
dataset = open(data_path)
data = [eval(i) for i in dataset]

preprocess_result_path = create_result_preprocessing()


headlines = []
for i in data:
    headlines.append(i["headline"])

labels = []
for i in data:
    labels.append(i["is_sarcastic"])


#print("Getting term...")
#terms = set()

#for headline in headlines:
#    term = get_terms(headline, remove_digit=True)
#    terms.update(term)
#    print(len(terms))
#
#write_file(os.path.join(preprocess_result_path, "dictionary2.txt"), terms)


terms = file_to_set(os.path.join(preprocess_result_path, "dictionary2.txt"))


print("Making boolean model...")
for i in range(len(headlines)):
    token = get_terms(headlines[i])

    example = list()
    for i_ in terms:
        example.append(1 if (i_ in token) else 0)

    example.append(labels[i])

    with open(os.path.join(preprocess_result_path, "dataset2.csv"), "a+") as f:

    #with open(os.path.join(preprocess_result_path, "dataset.txt"), "a+") as f:
        f.write(str(example)+"\n")

