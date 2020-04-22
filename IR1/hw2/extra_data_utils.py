import torch
import torch.nn as nn
import os
import operator
import numpy as np
import scipy

from tqdm import tqdm
from collections import defaultdict, Counter
import pickle as pkl

def get_bow_matrix(docs_by_id, word2id, query_bool, query):
    """
    Manually create a BOW-matrix for the given dataset or query.
    """
    if query_bool == False:
        path = "./pickles/bow_term_doc_matrix.pkl"

        num_docs = len(docs_by_id.keys())
        num_words = len(word2id.keys())

        # empty list of lists of size (num_words X num_docs)
        term_doc_matrix = np.zeros((num_words, num_docs))

        #populate term_doc matrix with binary counts
        for i, key in enumerate(docs_by_id.keys()):
            doc = docs_by_id[key]
            for word in set(doc):
                if word in word2id.keys():
                    word_id = word2id[word]
                    term_doc_matrix[word_id][i] = 1
                else:
                    continue
            if i%10000 == 0:
                print('processed ' + str(i) + ' documents for term_doc matrix')

        print(term_doc_matrix.shape)

        term_doc_matrix = scipy.sparse.csr_matrix(term_doc_matrix)

        with open(path, "wb") as writer:
            pkl.dump(term_doc_matrix, writer)

        return term_doc_matrix

    elif query_bool == True:
        num_docs = 1
        num_words = len(word2id.keys())

        term_doc_matrix = np.zeros((num_words, num_docs))

        # populate term_doc matrix with binary counts
        for word in set(query):
            if word in word2id.keys():
                word_id = word2id[word]
                term_doc_matrix[word_id] = 1
            else:
                continue

        print(term_doc_matrix.shape)

        term_doc_matrix = scipy.sparse.csr_matrix(term_doc_matrix)

        return term_doc_matrix



def get_tfidf_matrix(docs_by_id, word2id, df_dic, query_bool, query):
    """
    Manually create a TF-IDF matrix for the given dataset or query.
    """
    if query_bool == False:
        path = "./pickles/tfidf_matrix.pkl"

        #initialize matrix of 0's of size (vocabulary X num_documents)
        num_docs = len(docs_by_id.keys())
        num_words = len(word2id.keys())
        tfidf_matrix = np.zeros((num_words, num_docs))

        for i, key in enumerate(docs_by_id.keys()):
            doc = docs_by_id[key]
            doc_length = len(doc)
            for term in set(doc):
                if term in word2id.keys():
                    #fetch document frequency, compute inverse document frequency
                    term_df = df_dic[term]
                    idf = np.log10(num_docs / float(term_df))

                    #count occurences of term in current doc, compute term frequency and tfidf. populate tfidf_matrix with scores.
                    term_count = doc.count(term)
                    term_freq = term_count / doc_length
                    tfidf_score = term_freq * idf
                    term_id = word2id[term]
                    tfidf_matrix[term_id][i] = tfidf_score
                else:
                    continue
            if i%10000 == 0:
                print('processed ' + str(i) + ' documents for tfidf matrix')


        print(tfidf_matrix.shape)
        print(tfidf_matrix.nbytes)

        tfidf_matrix = scipy.sparse.csr_matrix(tfidf_matrix)

        with open(path, "wb") as writer:
            pkl.dump(tfidf_matrix, writer)

        return tfidf_matrix
    elif query_bool == True:

        # initialize matrix of 0's of size (vocabulary X num_documents)
        num_docs = 1
        num_words = len(word2id.keys())
        tfidf_matrix = np.zeros((num_words, num_docs))

        doc_length = len(query)
        for term in set(query):
            if term in word2id.keys():
                # fetch document frequency, compute inverse document frequency
                term_df = df_dic[term]
                idf = np.log10(num_docs / float(term_df))

                # count occurences of term in current doc, compute term frequency and tfidf. populate tfidf_matrix with scores.
                term_count = query.count(term)
                term_freq = term_count / doc_length
                tfidf_score = term_freq * idf
                term_id = word2id[term]
                tfidf_matrix[term_id] = tfidf_score
            else:
                continue
        print(tfidf_matrix.shape)
        tfidf_matrix = scipy.sparse.csr_matrix(tfidf_matrix)
        return tfidf_matrix

