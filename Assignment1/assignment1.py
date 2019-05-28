import sys
import nltk
from nltk.corpus import stopwords
from collections import Counter
import string
import pandas as pd
import numpy as np
from random import shuffle
import csv

import sys
import nltk
from nltk.corpus import stopwords
from collections import Counter
import string
import pandas as pd
import numpy as np
from random import shuffle
import csv
special = '!"#$%&()*+/:;<=>@[\\]^`{|}~\t\n'
def get_tokens(text):
    lowers = text.lower()
    remove_punctuation_map = dict((ord(char), ' '+char+' ') for char in string.punctuation+special)
    punctuation = lowers.translate(remove_punctuation_map)
    tokens = punctuation.split()
    return tokens


def cutPart(text):
    tokens = get_tokens(str(text))
    #remove special charactor
    tokens = [word for word in tokens if word not in special+', ']
    #remove stop words
    filtered = [w for w in tokens if not w in stopwords.words('english')]
    return tokens, filtered


def splitData(data):
    shuffle(data)
    length = len(data)
    segement1 = int(0.8 * length)
    segement2 = int(0.9 * length)
    train = np.array(data[:segement1])
    validation = np.array(data[segement1:segement2])
    test = np.array(data[segement2:])
    return train, validation, test

def readtxt(fileName):
    with open(fileName, 'r') as f:
        return [line for line in f.readlines()]

if __name__ == "__main__":
    input_path = sys.argv[1]
    text = readtxt(input_path)
    print('clean data')
    no_stop_word = [None] * len(text)
    with_stop_word = [None] * len(text)
    for i in range(len(text)):
        with_stop_word[i], no_stop_word[i] = cutPart(text[i])
    train_with_stop ,vali_with_stop, test_with_stop = splitData(with_stop_word)
    train_no_stop ,vali_no_stop, test_no_stop = splitData(no_stop_word)

    # sample_tokenized_list = [["Hello", "World", "."], ["Good", "bye"]]

    np.savetxt(input_path[-7:-4]+"_train.csv", train_with_stop, delimiter=",", fmt='%s')
    np.savetxt(input_path[-7:-4]+"_val.csv", vali_with_stop, delimiter=",", fmt='%s')
    np.savetxt(input_path[-7:-4]+"_test.csv", test_with_stop, delimiter=",", fmt='%s')

    np.savetxt(input_path[-7:-4]+"_train_no_stopword.csv", train_no_stop,
               delimiter=",", fmt='%s')
    np.savetxt(input_path[-7:-4]+"_val_no_stopword.csv", vali_no_stop,
               delimiter=",", fmt='%s')
    np.savetxt(input_path[-7:-4]+"_test_no_stopword.csv", test_no_stop,
               delimiter=",", fmt='%s')