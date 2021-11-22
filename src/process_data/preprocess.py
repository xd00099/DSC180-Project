import pandas as pd
import numpy as np
import re
import os
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import pickle
import nltk

YEAR_THRESHOLD = 2015

def lemmatize_stemming(text):
    return WordNetLemmatizer().lemmatize(text, pos='v')

def preprocess_abstract(text):
    result = []
    redundant = ['abstract', 'purpose', 'paper', 'goal']
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in redundant:
            result.append(lemmatize_stemming(token))
    return " ".join(result)


def clean(data_path):
    data = pd.read_csv(data_path, index_col=0)
    data = data.fillna('')

    
    stemmer = PorterStemmer()

    data['abstract_processed'] = data['abstract'].apply(preprocess_abstract)
    
    data['year'] = data['year'].astype(int)
    data = data[data['year'] > YEAR_THRESHOLD]
    

    # organzie author's abstracts by year
    authors = {}
    for author in data['HDSI_author'].unique():
        authors[author] = {
            2016 : list(),
            2017 : list(),
            2018 : list(),
            2019 : list(),
            2020 : list(),
            2021 : list()
        }
    for i, row in data.iterrows():
        authors[row['HDSI_author']][row['year']].append(row['abstract_processed'])

    all_docs = []
    for author, author_dict in authors.items():
        for year, documents in author_dict.items():
            all_docs.append(" ".join(documents))

    return all_docs, authors

def save_cleaned_corpus(data_path, output_path_corpus, output_path_authors):
    corpus, authors = clean(data_path)
    pickle.dump(corpus, open(output_path_corpus, 'wb'))
    pickle.dump(authors, open(output_path_authors, 'wb'))