import torch
import sys
import pickle as pkl
import argparse
import data_utils
import torch.nn.functional as F
import word2vec_model
import operator
import heapq
import sklearn
import os
import numpy as np
import read_ap as ra
import pytrec_eval
import json
from sklearn.metrics.pairwise import cosine_similarity
from torch.autograd import Variable
from torch import nn
from tqdm import tqdm


def compute_metrics(docs, vocab_embs, word2id, id2word):
    """
    For a trained model, compute the MAP and NDCG based on a set of queries and
    all documents in the corpus.

    Returns:
        metrics: a nested dict of queries and their MAP and NDCG scores.
    """
    # Create document embeddings
    if not os.path.exists("./pickles/word2vec_doc_embs.pkl"):
        print("constructing document embeddings")
        doc_embs = {}
        keys = list(docs.keys())
        for d in tqdm(keys):
            doc = docs[d]
            doc_emb = create_doc_emb(vocab_embs, doc, word2id, id2word)
            doc_embs[d] = doc_emb

        with open("./pickles/word2vec_doc_embs.pkl", "wb") as writer:
                pkl.dump(doc_embs, writer)
    else:
        with open("./pickles/word2vec_doc_embs.pkl", "rb") as reader:
            doc_embs = pkl.load(reader)

    # Create query embedding and compare to every docuemnt embedding
    qrels, queries = ra.read_qrels()
    overall_ser = {} #ranking per query
    for qid in tqdm(qrels):
        query = queries[qid]
        query = ra.process_text(query)
        query_emb = create_doc_emb(vocab_embs, query, word2id, id2word)
        ranking, trec_results = get_ranking(qid, query_emb, doc_embs, vocab_embs)
        overall_ser[qid] = ranking

        if not int(qid) in range(76,100):
            with open("./results/word2vec_trec.csv", "a+") as f:
                f.write("\n".join("{},{},{},{},{},{}".format(x[0], x[1],x[2],x[3],x[4],x[5]) for x in trec_results))
                f.write("\n")

    # Compute the MAP and NDCG per query
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})
    metrics = evaluator.evaluate(overall_ser)

    # Get the average model evaluation scores over all queries
    average = {'map':0, 'ndcg':0}
    for q in list(metrics.values()):
        average['map'] += q['map']
        average['ndcg'] += q['ndcg']
    average['map'] = average['map']/len(queries)
    average['ndcg'] = average['ndcg']/len(queries)
    print('average model evaluation scores over all queries {}'.format(average))

    return(metrics)


def get_ranking(qid, query_emb, doc_embs, vocab_embs):
    """
    Given a query embedding, compute the cosine similarity score
    for every document in a dict of document embeddings.

    Returns:
        ranking: sorted dict (doc_id, score)
        trec_results: list of tuples with all TREC values per doc
    """
    scores = {}
    keys = list(doc_embs.keys())

    # Compare the query embedding to every document embedding and save the score
    for d in (keys):
        doc_emb = doc_embs[d]
        score = sklearn.metrics.pairwise.cosine_similarity(query_emb, doc_emb, dense_output=True).item()
        scores[d] = score

    # Sort scores from high to low
    ranking = dict(sorted(scores.items(), key=operator.itemgetter(1), reverse=True))

    # Convert dict to list of tuples and get TREC values per document
    ranking_tuples = ranking.items()
    trec_results = [(qid,"") + (tup[0],) + (i,) + (tup[1],) + ('word2vec',) for i,tup in enumerate(ranking_tuples)]
    return(ranking, trec_results)

def create_doc_emb(matrix, doc, word2id, id2word):
    '''
    Takes a list of words, converts these words to id's, computes the word
    embedding for each word and sums these embeddings to get the
    document representation

    Returns:
        doc_emb: array [emb_dim]
    '''
    embeddings = []
    for word in doc:
        word = ra.process_text(word)
        if len(word) == 1:
            word_id = word2id.get(word[0])
            if word_id != None:
                word_embedding = matrix[word_id,:].reshape(1,-1)
                embeddings.append(word_embedding)
    embeddings = np.asarray(embeddings)
    doc_emb = embeddings.mean(axis=0)
    return doc_emb


def similar_words(vocab_embs, word, n_of_similar_words, word2id, id2word):
    '''
    Takes a word, gets the corresponding word embedding, and computes the
    cosine similarity score with the embeddings of all other words in the vocab.

    Returns:
        similar words: list of n most similar words.
    '''
    word = ra.process_text(word)[0]
    word_id = word2id[word]
    word_emb = vocab_embs[word_id,:].reshape(1, -1)

    # Compute cosine similarity score for every word in the vocabulary
    scores = sklearn.metrics.pairwise.cosine_similarity(word_emb, vocab_embs, dense_output=True)
    scores = list(scores[0])

    # Get n words with highest scores
    best_n = heapq.nlargest(n_of_similar_words, range(len(scores)), scores.__getitem__)
    similar_words = []
    for id in best_n:
        similar_words.append(id2word[id])

    return(similar_words)
def train_word2vec(train_corpus, word2id, id2word):
    '''
    Takes a concatenaded list of all words in the corpus and trains an embedding
    matrix consisting of word embeddings for each word in the vocabulary.

    Returns:
        vocab_emb: array [vocab_size, emb_dim]
    '''
    vocab_size = len(word2id)
    model = word2vec_model.Word2VecModel(vocab_size=vocab_size, embedding_dim=ARGS.embedding_dim)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr = ARGS.learning_rate)

    for iter in range(ARGS.iterations):
        # get batch
        t, cp, cn = data_utils.get_word2vec_batch(train_corpus, ARGS.batch_size, ARGS.window, ARGS.n_neg_samples, vocab_size)

        optimizer.zero_grad()
        loss = model.forward(t, cp, cn)
        loss.backward()
        optimizer.step()

        if iter%100 == 0:
            # Save temporary vocabulary embeddings
            vocab_embs = model.t_vocab_embs.weight.data.numpy()
            with open("./models/word2vec/iter_{}.pkl".format(iter), "wb") as writer:
                    pkl.dump(vocab_embs, writer)

            # Most similar words to 'president'
            sim_words = similar_words(vocab_embs, 'war', 10, word2id, id2word)

            print("--- iteration {}: loss {} similar words {}".format(iter, round(loss.item(),5), sim_words))

    # Save final vocabulary embeddings
    vocab_embs = model.t_vocab_embs.weight.data.numpy()
    with open("./models/word2vec/final_model.pkl", "wb") as writer:
            pkl.dump(vocab_embs, writer)

    return(vocab_embs)

def data_loader():
    '''
    Loads the documents (by id in a dict and concatenated in a list) and the
    word2id/id2word dicts.
    '''
    # Load documents
    if not os.path.exists("./pickles/processed_docs.pkl"):
        docs_by_id = ra.get_processed_docs()
    else:
        with open("./pickles/processed_docs.pkl", "rb") as reader:
            docs_by_id = pkl.load(reader)

    # Load word2id and id2word documents
    if not os.path.exists("./pickles/word2id.pkl"):
        print("constructing word2id and id2word dicts")
        word2id, id2word = data_utils.counter_to_dicts(docs_by_id)
    else:
        with open("./pickles/word2id.pkl", "rb") as reader:
            word2id = pkl.load(reader)
        with open("./pickles/id2word.pkl", "rb") as reader:
            id2word = pkl.load(reader)


    # Load word2vec corpus
    if not os.path.exists("./pickles/word2vec_corpus.pkl"):
        print("creating train_corpus")
        word2vec_corpus = data_utils.create_word2vec_corpus(docs_by_id, word2id)
    else:
        with open("./pickles/word2vec_corpus.pkl", "rb") as reader:
            word2vec_corpus = pkl.load(reader)

    return(word2id, id2word, word2vec_corpus, docs_by_id)



def main():
    # Load data
    word2id, id2word, word2vec_corpus, docs_by_id = data_loader()

    #Train model and get vocab embeddings
    if not os.path.exists("./models/word2vec/final_model.pkl"):
        print("training model")
        vocab_embs = train_word2vec(word2vec_corpus, word2id, id2word)
    else:
        with open("./models/word2vec/final_model.pkl", "rb") as reader:
            vocab_embs = pkl.load(reader)

    # Write TREC results column headers to file
    if not os.path.exists("./results/word2vec_trec.csv"):
        with open("./results/word2vec_trec.json", "w") as f:
            f.write("query-id, Q0, document-id, rank, score, STANDARD\n")

    # Compute evaluation scores for the model
    print("evaluating model")
    metrics = compute_metrics(docs_by_id, vocab_embs, word2id, id2word)
    with open("./results/word2vec.json", "w") as writer:
        json.dump(metrics, writer, indent=1)

# -----------------------------------------------------------
# Compile ARGS and run main()
# -----------------------------------------------------------

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()

    PARSER.add_argument('--embedding_dim', default=200, type=int, #200
                            help='dimension of embedding matrix')
    PARSER.add_argument('--iterations', default=200000, type=int,
                        help='max number of iterations')
    PARSER.add_argument('--batch_size', default=1024, type=int,
                        help='size of batch')
    PARSER.add_argument('--window', default=10, type=int, #10
                        help='size of context window')
    PARSER.add_argument('--n_neg_samples', default=10, type=int,
                        help='number of negative samples per positive pair')
    PARSER.add_argument('--learning_rate', default=2e-3, type=int,
                        help='learning rate of the optimizer')

    ARGS = PARSER.parse_args()


    main()
