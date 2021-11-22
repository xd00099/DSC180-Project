import pandas as pd
import numpy as np
import re
import os
import pickle
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
np.random.seed(123)


def train_lda_model(corpus_path, authors_path, k):

    with open(corpus_path, 'rb') as fp:
        all_docs = pickle.load(fp)

    with open(authors_path, 'rb') as fp:
        authors = pickle.load(fp)

    # initate LDA model
    countVec = CountVectorizer()
    counts = countVec.fit_transform(all_docs)
    names = countVec.get_feature_names()


    # k topics model 
    lda = LatentDirichletAllocation(n_components=k, n_jobs=-1, random_state=123)
    model_lda = lda.fit_transform(counts)

    # column names
    topicnames = ["Topic" + str(i) for i in range(k)]

    # index names
    docnames = ["Doc" + str(i) for i in range(len(all_docs))]

    # Make the pandas dataframe
    df_document_topic = pd.DataFrame(model_lda, columns=topicnames, index=docnames)

    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic

    # add author and year
    df_document_topic['author'] = np.nan
    df_document_topic['year'] = np.nan
    df_document_topic.shape

    year_paper_count = {}
    for author in authors.keys():
        if author not in year_paper_count.keys():
            year_paper_count[author] = 0
        year_paper_count[author] += len(authors[author])

    author_list = list(year_paper_count.keys())
    author_list_populate = np.array([[a]*6 for a in author_list]).flatten()
    df_document_topic.iloc[:, k+1] = author_list_populate

    year = [2016, 2017, 2018, 2019, 2020, 2021] * len(author_list)
    df_document_topic.iloc[:, k+2] = year

    time_author_topic = df_document_topic

    author_topic = time_author_topic.groupby('author').agg({'dominant_topic': lambda x:x.value_counts().index[0]})

    return model_lda, df_document_topic, time_author_topic, author_topic


def save_lda_model(corpus_path, authors_path, k, model_output_path, dominant_topic_output, time_author_topic_output, author_topic_output):
    model_lda, df_document_topic, time_author_topic, author_topic = train_lda_model(corpus_path, authors_path, k)
    pickle.dump(model_lda, open(model_output_path, 'wb'))
    df_document_topic.to_csv(dominant_topic_output)
    time_author_topic.to_csv(time_author_topic_output)
    author_topic.to_csv(author_topic_output)
    