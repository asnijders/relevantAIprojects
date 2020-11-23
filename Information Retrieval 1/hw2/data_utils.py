import torch
import torch.nn as nn
import os
import operator
import numpy as np
import scipy

from tqdm import tqdm
from collections import defaultdict, Counter
import pickle as pkl

import read_ap as ra
import random
import sys
import itertools


def create_word2vec_corpus(docs_by_id, word2id):
    '''
    Concatenates all the documents to one list of words.
    '''
    path = "./pickles/word2vec_corpus.pkl"

    list_of_lists = list(docs_by_id.values()) # make list of values
    flat_list = list(itertools.chain(*list_of_lists)) # flatten the list

    word2vec_corpus = []
    # Substitute words by their corresponding id
    for i, word in enumerate(flat_list):
        word2vec_corpus.append(word2id.get(word))

    with open(path, "wb") as writer:
            pkl.dump(word2vec_corpus, writer)

    return word2vec_corpus

def create_counter(docs_by_id):
    """
    Creates a dictionary with all unique words in the dataset and
    their occurence counts.
    """
    path = "./pickles/word_counts.pkl"
    doc_ids = list(docs_by_id.keys())
    count_dict = defaultdict(list)
    for doc_id in tqdm(doc_ids):
        doc = docs_by_id[doc_id]

        counts = Counter(doc)
        for (t, c) in counts.items():
            count_dict[t].append(c)

    final_dict = {k:sum(v) for k,v in count_dict.items()}

    with open(path, "wb") as writer:
            pkl.dump(final_dict, writer)

    return final_dict


def counter_to_dicts(docs_by_id):
    """
    Create two dictionaries that map all unique words that occur more than
    50 times in the dataset to an id and vice versa.
    """
    count_dict = create_counter(docs_by_id)
    path1 = "./pickles/word2id.pkl"
    path2 = "./pickles/id2word.pkl"
    word2id, id2word = {}, {}

    # Create dics with only words with a fruency of at least 50
    counter = 0
    for word, count in count_dict.items():
        if count > 50:
            word2id[word] = counter
            id2word[counter] = word
            counter += 1

    # Save dics
    with open(path1, "wb") as writer:
            pkl.dump(word2id, writer)
    with open(path2, "wb") as writer:
            pkl.dump(id2word, writer)

    return word2id, id2word


def get_doc_freqs(docs_by_id):
    """
    Get the frequencies of unique words per doc.
    """
    path = "./pickles/document_freqs.pkl"

    df_dic = {}
    for key in docs_by_id.keys():
        document = docs_by_id[key]
        for term in set(document):
            if term not in df_dic.keys():
                df_dic[term] = 1
            else:
                df_dic[term] += 1

    with open(path, "wb") as writer:
        pkl.dump(df_dic, writer)

    return df_dic

def get_word2vec_batch(corpus, batch_size, window, n_neg_samples, neg_samples_max):
    '''
    Samples a batch by starting at a random target word in a list of word id's
    and constructing pairs with the surrounding context words. Also,
    for each pair, a number of random id's are used as negative samples and
    stored in a matrix.

    Returns:
    t: list of target word id's [batch_size]
    cp: lit of context word id's [batch_size]
    cn: matrix of random id's [batch_size, n_neg_samples]
    '''
    window_lr = int(((window-1)/2)) #windows size left and right of target
    t = []
    cp = []
    cn = []

    start = random.randint(0, len(corpus))

    for i, target in enumerate(corpus[start:]): # loop over targets
        for j, context in enumerate(corpus[max(0,start+i-window_lr):min(start+i+window_lr+1, len(corpus))]): # loop over window around target
            if ((target != context) and (target != None) and (context != None)):
                t.append(target)
                cp.append(context)
            if len(t) == batch_size:
                break
        if len(t) == batch_size:
            break

    cn = torch.randint(0, neg_samples_max, (batch_size, n_neg_samples))

    return(t, cp, cn)
