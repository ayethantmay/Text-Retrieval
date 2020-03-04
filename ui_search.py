from flask import Flask, render_template, redirect, url_for, request
app = Flask(__name__)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from num2words import num2words
from itertools import chain
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
import os
import string
import numpy as np
import copy
import pandas as pd
import pickle
import re
import math


@app.route('/')
def index():
   return render_template("input.html")



result = []
dataset = []
detail = []

@app.route('/20newsgroup',methods = ['POST'])
def newsgroup():

   folders = ['20news-18828/rec.motorcycles','20news-18828/rec.sport.baseball','20news-18828/rec.sport.hockey','20news-18828/sci.space','20news-18828/soc.religion.christian']
   filenames = [os.listdir(f) for f in folders]
   fil_dict = dict(zip(folders, filenames))
   #print("test",fil_dict)
   for (dict_key, files_list) in fil_dict.items():
      for filename in files_list:
         dataset.append((dict_key + '/' + filename))
   N = len(dataset)
   

   #Preprocessing
   def convert_lower_case(data):
      return np.char.lower(data)

   def remove_stop_words(data):
      stop_words = stopwords.words('english')
      words = word_tokenize(str(data))
      new_text = ""
      for w in words:
         if w not in stop_words and len(w) > 1:
               new_text = new_text + " " + w
      return new_text


   def remove_punctuation(data):
      symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
      for i in range(len(symbols)):
         data = np.char.replace(data, symbols[i], ' ')
         data = np.char.replace(data, "  ", " ")
      data = np.char.replace(data, ',', '')
      return data


   def remove_apostrophe(data):
      return np.char.replace(data, "'", "")


   def stemming(data):
      stemmer= PorterStemmer()
      tokens = word_tokenize(str(data))
      new_text = ""
      for w in tokens:
         new_text = new_text + " " + stemmer.stem(w)
      return new_text


   def convert_numbers(data):
      tokens = word_tokenize(str(data))
      new_text = ""
      for w in tokens:
         try:
               w = num2words(int(w))
         except:
               a = 0
         new_text = new_text + " " + w
      new_text = np.char.replace(new_text, "-", " ")
      return new_text


   def preprocess(data):
      data = convert_lower_case(data)
      data = remove_punctuation(data) 
      data = remove_apostrophe(data)
      data = remove_stop_words(data)
      data = convert_numbers(data)
      data = stemming(data)
      data = remove_punctuation(data)
      data = convert_numbers(data)
      data = stemming(data) 
      data = remove_punctuation(data) 
      data = remove_stop_words(data) 
      return data



   #Extracting processed data
   processed_text = []
   for i in dataset:
      file = open(i, 'r', encoding="utf8", errors='ignore')
      text = file.read().strip()
      file.close()
      processed_text.append(word_tokenize(str(preprocess(text))))




   #Calculating DF for all words
   DF = []
   d = {}

   for i in range(N):
      tokens = processed_text[i]
      for w in tokens:
         d = {'key1':'pass'}
         d[w] = d.pop('key1')
         DF.append(d)

   j = Counter(chain.from_iterable(j.keys() for j in DF))
   DF = dict(j)
   #print(DF)

   total_vocab_size = len(DF)
   #print(DF) #'{buddihsm': 1, 'ara': 1, 'disserv': 1}
   #print(total_vocab_size) #32184
   total_vocab = [x for x in DF]



   def doc_freq(word):
    c = 0
    try:
        c = DF[word]
    except:
        pass
    return c


   #Calculating TF-IDF for text
   doc = 0
   tf_idf = {}
   for i in range(N):

      tokens = processed_text[i]
      counter = Counter(tokens) #c = Counter([1, 2, 21, 12, 2, 44, 5, 13, 15, 5, 19, 21, 5]) // output c = 1 2 2 19 21 21 44 15 12 13 5 5 5 
      
      for token in np.unique(tokens):
         
         tf = np.log(1+counter[token])
         df = doc_freq(token)
         #print(df) # 1,1,1,...
         idf = np.log((N+1)/(df+1))
         
         tf_idf[doc, token] = tf*idf # (51, 'throttl'): 5.419322899490301
         #print("tf_idf", tf_idf)

      doc += 1


   #Vectorising tf-idf (indexing)
   D = np.zeros((N, total_vocab_size))
   for i in tf_idf:
      try:
         #print(i) #(81, 'speed'): 8.589423574991576
         ind = total_vocab.index(i[1])
         D[i[0]][ind] = tf_idf[i]
      except:
         pass





   #TF-IDF Cosine Similarity Ranking
   def cosine_similarity(a, b):
      cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
      return cos_sim



   def vectorize(tokens):
      Q = np.zeros((len(total_vocab)))
      counter = Counter(tokens)
      
      for token in np.unique(tokens):
         
         tf = math.log(1+counter[token])
         df = doc_freq(token)
         idf = math.log((N+1)/(df+1))

         try:
               ind = total_vocab.index(token)
               Q[ind] = tf*idf
         except:
               pass
      return Q


   def retrieve(k, query):

      preprocessed_query = preprocess(query)
      tokens = word_tokenize(str(preprocessed_query))
      
      print("\nQuery:", query)
      print("")
      print("\nTokens from Query:", tokens)
      
      d_cosines = []
      
      query_vector = vectorize(tokens)
      
      for d in D:
         d_cosines.append(cosine_similarity(query_vector, d))
         
      out = np.array(d_cosines).argsort()[-k:][::-1]
      
      print("")
      
      print("\nTop 10 relevant documents:", out)
      return out
      


 

   if request.method == 'POST':
      query = request.form['q']
      Q = retrieve(10, query)
      for i in Q:
         file = open(dataset[i], 'r', encoding='cp1250')
         text = file.read().strip()
         file.close()
         detail.append(text)
         result.append(dataset[i])
      print(result)
      return redirect(url_for('success' ))
      

@app.route('/success')
def success():
   return render_template('output.html', name1 = result[0], name2 = result[1], name3 = result[2], 
   name4 = result[3], name5 = result[4], name6 = result[5], name7 = result[6], name8 = result[7],
   name9 = result[8], name10 = result[9] )

@app.route('/page1')
def page1():
   return render_template('details.html', name = detail[0])
      
@app.route('/page2')
def page2():
   return render_template('details.html', name = detail[1])

@app.route('/page3')
def page3():
   return render_template('details.html', name = detail[2])

@app.route('/page4')
def page4():
   return render_template('details.html', name = detail[3])

@app.route('/page5')
def page5():
   return render_template('details.html', name = detail[4])

@app.route('/page6')
def page6():
   return render_template('details.html', name = detail[5])

@app.route('/page7')
def page7():
   return render_template('details.html', name = detail[6])

@app.route('/page8')
def page8():
   return render_template('details.html', name = detail[7])

@app.route('/page9')
def page9():
   return render_template('details.html', name = detail[8])

@app.route('/page10')
def page10():
   return render_template('details.html', name = detail[9])

   
if __name__ == '__main__':
   app.run(debug = True)