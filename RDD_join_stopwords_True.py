#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 10:05:46 2019

@author: Arnaud
"""


#RDD join - stopwords cleaning

data_text_files = sc.wholeTextFiles("bbc/business/", 8) #loading sets of files in the folder bbc
#cleaning function - remove digits and punctuation and put everything in lowercase
def lower_clean_str(x):
  punc='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
  digit='0123456789'
  lowercased_str = x.lower()
  
  for ch in punc:
    lowercased_str = lowercased_str.replace(ch, '')
  for dig in digit:
    lowercased_str = lowercased_str.replace(dig, '')
  return lowercased_str.replace('\n', '')


def stopword(x):
    stopwords= ['a','able','about','across','after','all','almost','also','am','among','an','and','any','are','as','at','be','because','been','but','by',
            'can','cannot','could','dear','did','do','does','either','else','ever','every','for','from','get','got','had','has','have','he','her','hers',
            'him','his','how','however','i','if','in','into','is','it','its','just','least','let','like','likely','may','me','might','most','must','my',
            'neither','no','nor','not','of','off','often','on','only','or','other','our','own','rather','said','say','says','she','should','since','so',
            'some','than','that','the','their','them','then','there','these','they','this','tis','to','too','twas','us','wants','was','we','were','what',
            'when','where','which','while','who','whom','why','will','with','would','yet','you','your' 'bn','£bn', 'see','£m']
    
    x = [ value for value in x if value not in stopwords]
    return x


data_text_cleared = data_text_files.mapValues(lower_clean_str)

#tf part - count number of word occurence in each doc
split_data = data_text_cleared.mapValues(lambda x: x.split()) #first, we split the data at every ' '
split_data_cleared = split_data.mapValues(stopword)

term_freq = split_data_cleared.flatMapValues(lambda x: x).map(lambda x: ((x[0],x[1]), 1)) #apply a flatmap 
# and create a map with key (filename, word) and value 1
rdd_term_freq = term_freq.reduceByKey(lambda x,y: x+y) #output is a RDD with Key (Title, word) and Value: nb_occurence


doc_freq = split_data.flatMapValues(lambda x: x).distinct() # get (file, word) for evry time a word is inside a doc
idf_rdd = doc_freq.map(lambda x: (x[1],1)).reduceByKey(lambda x,y: x+y) #we have a similar output as in idf_dict but with a rdd 

#using Spark join, we first need to change the Key of rdd_term_freq to have key: Word and value: (file, nb_occurence)
joined_rdd = rdd_term_freq.map(lambda x: ((x[0][1]), (x[0][0], x[1]))).join(idf_rdd)
#output is rdd (word, (( file, tf), rdf))

tf_idf_dict_join = joined_rdd.map(lambda x: ((x[1][0][0], x[0]) , x[1][0][1]/x[1][1])).collectAsMap()