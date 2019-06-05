import sys
import string
import pandas as pd
import numpy as np
from random import shuffle
import csv
import string
#nltk.download('stopwords')
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
def readtxt(fileName):
    with open(fileName, 'r') as f:
        return [ast.literal_eval(line) for line in f.readlines()]

def dummy(doc):
    return doc
def remove_punt(doc):
  doc = [word for word in doc if word.isalpha()]
  return doc
def n_gram(text, lower,  upper):
  vectorizer = TfidfVectorizer(ngram_range = (lower, upper),tokenizer=dummy, preprocessor=remove_punt)
  bagofword = vectorizer.fit_transform(text)
  bagofword = bagofword
  return bagofword, vectorizer


def preprocessing(trainText,valiText, testText, n_lower, n_upper):
  train_tfbags, model = n_gram(trainText, n_lower, n_upper)
  vali_tfbags = model.transform(valiText)
  test_tfbags = model.transform(testText)  
  vocab = model.get_feature_names()
  return train_tfbags, vali_tfbags, test_tfbags, vocab

def tuning(trainX,trainY,valiX, valiY):
  accuracy = []
  alphas = np.arange(0.01, 1.11, 0.1)
  for alpha in alphas:
    accuracy.append(classfier(trainX,trainY,valiX, valiY, alpha))
   
  bestAlpha  = alphas[np.argmax(accuracy)]
  return bestAlpha
  
def classfier(trainX, trainY, testX, testY, alpha):
  clf = MultinomialNB(alpha = alpha)
  clf.fit(trainX, trainY)
  return clf.score(testX, testY)

def applyToTest(trainX, trainY, testX, testY, bestAlpha):
  accuracy  = classfier(trainX, trainY, testX, testY, bestAlpha)
  return accuracy
def concat(pos,neg):
  labelpos = [1] * len(pos)
  labelneg = [0] * len(neg)
  x = pos + neg
  y = labelpos + labelneg
  return x, y
   

if __name__ == "__main__":
    #text = readtxt('drive/My Drive/Colab Notebooks/pos.txt')
    training_pos = sys.argv[1]
    training_neg = sys.argv[2]
    validation_pos = sys.argv[3]
    validation_neg = sys.argv[4] 
    test_pos = sys.argv[5]
    test_neg = sys.argv[6]
    training_pos = readtxt(training_pos)
    training_neg = readtxt(training_neg)
    validation_pos = readtxt(validation_pos)
    validation_neg = readtxt(validation_neg)
    test_pos = readtxt(test_pos)
    test_neg = readtxt(test_neg)
    trainX, trainY = concat(training_pos,training_neg)
    valiX, valiY = concat(validation_pos,validation_neg)
    testX, testY = concat(test_pos,test_neg)
    n_range = [(1,1), (2,2),(1,2)]
    for lower, upper in n_range:
      train_tfbags, vali_tfbags, test_tfbags, vocab = preprocessing(trainX, valiX, testX, lower , upper)
      bestAlpha = tuning(train_tfbags,trainY,vali_tfbags, valiY)
      print('best aplha is ' + str(bestAlpha))
      accuracy = applyToTest(train_tfbags, trainY, test_tfbags, testY, bestAlpha)
      print('accuracy for n range '+ str((lower, upper) + ' is ' + str(accuracy)))
      
    
    
    
    