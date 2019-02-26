#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 10:01:50 2019

@author: Arnaud
"""

#Alexis - Annabelle, voilà les fichiers python à run.
# Vous devenez d'abord mettre les fichier .txt (de la bbc) sur le cluster
#Il faut lancer Pyspark sur le cluster en lancant la commande Pyspark
#Ensuite faites un "copy-paste" du code en dessous EN CHANGEANT LE PATH DU FICHIER
#il faut ajouter un time checker


#Broadcast dict - stopwords cleaning

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
idf_dict = doc_freq.map(lambda x: (x[1],x[0])).countByKey() # invert key value in the dictionnary for later - output is a dict as we apply the action .countByKey

idf_score = sc.broadcast(dict(idf_dict)) #using broadcasting variable for a map join 

# Creaing a RDD with key (file, word) and value (tf, idf)
tf_idf_raw = rdd_term_freq.map(lambda x: ((x[0][0],x[0][1]), (x[1], idf_score.value[x[0][1]])))
#brodcasting the number of element in the corpus
nb_doc = data_text_files.count() #counting the number of files
doc_nb = sc.broadcast(nb_doc)
tf_idf_dict = tf_idf_raw.map(lambda x: (x[0], (x[1][0] * m.log10(doc_nb.value/x[1][1])))) #get the tf-idftf_idf_dict_broadcast = tf_idf_dict.collectAsMap() #RDD as a dict
tf_idf_dict_broadcast = tf_idf_dict.collectAsMap() #RDD as a dict
