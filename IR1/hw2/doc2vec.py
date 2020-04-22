import os
import sys
import argparse

import gensim
from gensim.models.callbacks import CallbackAny2Vec

import numpy as np
import read_ap as ra
from tqdm import tqdm
from collections import Counter
import pytrec_eval
import json
import pandas as pd


class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''
    def __init__(self):
        self.epoch = 0
    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))
    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1


def create_corpus(docs_by_id):
    """ 
    Create a gensim TaggedDocument corpus from the AP news data.
    """
    corpus = []
    for key, doc in docs_by_id.items():
        corpus.append(gensim.models.doc2vec.TaggedDocument(doc, [key]))
    return corpus


def train_doc2vec(train_corpus):
    """
    Train the gensim doc2vec model. 
    """
    epoch_logger = EpochLogger()

    # Declare model variables
    vec_dim = config.vec_dim
    win_size = config.win_size
    vocab_size = config.vocab_size
    print("Vector dimension:", vec_dim, "window size:", win_size, "Vocabulary size:", vocab_size)

    # Build and train the model
    model = gensim.models.doc2vec.Doc2Vec(vector_size=vec_dim, window=win_size, max_vocab_size=vocab_size, epochs=20)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs, callbacks=[epoch_logger])

    # Save the trained model to a file
    save_path = "./doc2vec_models/doc2vec_{}_{}_{}.model".format(vec_dim, win_size, vocab_size)
    model.save(save_path)

    return model


def rank_docs(model, query, qid, run_id):
    """
    Use a trained model to return the most similar docs to a query.
    
    Returns:
        sims: list of tuples (doc_id, score)
        trec_results: list of tuples with all TREC values per doc
    """
    query = ra.process_text(query)
    query_vec = model.infer_vector(query, epochs=200)
    sims = model.docvecs.most_similar([query_vec], topn=len(model.docvecs))
    trec_results = [(qid,"") + (tup[0],) + (i,) + (tup[1],) + (run_id,) for i,tup in enumerate(sims)]
    return sims, trec_results

def test_doc(filename):
    """
    Convert txt file document to query format.
    """
    query = ""
    with open(filename, "r") as reader:
        for line in reader.readlines():
            line = str(line)
            query += line
    return query

def main():
    docs_by_id = ra.get_processed_docs()
    path = "./doc2vec_models/{}".format(config.model_name)
    # print(path)
    if not os.path.exists(path):
        print("Model not yet trained, starting training now.")
        train_corpus = create_corpus(docs_by_id)
        model = train_doc2vec(train_corpus)
    else:
        print("Model already trained, loading the file.")
        model = gensim.models.doc2vec.Doc2Vec.load(path)

    qrels, queries = ra.read_qrels()
    print(queries)

    overall_ser = {}
    trec_path = "./results/trec_doc2vec.csv"

    # Write TREC results column headers to file
    with open(trec_path, "w") as f:
        f.write("query-id, Q0, document-id, rank, score, STANDARD\n")

    print("Evaluating doc2vec model:", config.model_name)

    # Loop over all queries and predict most relevant docs
    for qid in tqdm(qrels):
        query_text = queries[qid]
        results, trec_results = rank_docs(model, query_text, qid, config.model_name)
        results = dict(results)
        overall_ser[qid] = results
        # Write all test queries to TREC format file
        if not int(qid) in range(76,100):
            with open(trec_path, "a+") as f:
                f.write("\n".join("{},{},{},{},{},{}".format(x[0], x[1],x[2],x[3],x[4],x[5]) for x in trec_results))
                f.write("\n")

    # run evaluation with `qrels` as the ground truth relevance judgements
    # here, we are measuring MAP and NDCG
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})
    metrics = evaluator.evaluate(overall_ser)

    # dump this to JSON
    # *Not* Optional - This is submitted in the assignment!
    json_path = "./results/{}.json".format(config.model_name)
    with open(json_path, "w") as writer:
        json.dump(metrics, writer, indent=1)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--win_size', type=int, default=10,
                        help="context window size")
    parser.add_argument('--vec_dim', type=int, default=300,
                        help="dimension of the word embeddings")
    parser.add_argument('--vocab_size', type=int, default=50000,
                        help="max amount of unique words in vocab")
    parser.add_argument('--model_name', type=str, default="doc2vec_200_10_50000.model",
                        help="path to model that should be used")
    config = parser.parse_args()

    main()
