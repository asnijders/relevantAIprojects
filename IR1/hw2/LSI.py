import os
import argparse
from pprint import pprint
import pickle as pkl
import read_ap as ra
from tqdm import tqdm
from gensim import corpora
from gensim.models import LsiModel
from gensim.models import TfidfModel
from gensim import similarities
from read_ap import get_processed_docs
import operator
import pytrec_eval
import json

def create_corpus_and_dict(documents):
    """
    Retrieve all the necessary data to train the LSI model.
    """
    if not os.path.exists('./tmp/dictionary.dict'):
        print("Starting construction dictionary now")
        dictionary = corpora.Dictionary(documents)
        dictionary.save('./tmp/dictionary.dict')
    else:
        print("Dictionary already constructed, loading now...")
        dictionary = corpora.Dictionary()
        dictionary = dictionary.load('./tmp/dictionary.dict')

    #construct BOW corpus
    if not os.path.exists('./tmp/bow_corpus.mm'):
        print("Starting construction bow corpus now")
        bow_corpus = [dictionary.doc2bow(text) for text in documents]
        corpora.MmCorpus.serialize('./tmp/bow_corpus.mm', bow_corpus)
    else:
        print('BOW corpus already created, loading now...')
        bow_corpus = corpora.MmCorpus('./tmp/bow_corpus.mm')

    #construct TFIDF corpus
    if not os.path.exists('./tmp/tfidf_corpus.mm'):
        print("Starting construction TFIDF corpus now")
        corpus = [dictionary.doc2bow(text) for text in documents]
        model_tfidf = TfidfModel(corpus)
        tfidf_corpus = model_tfidf[corpus]
        corpora.MmCorpus.serialize('./tmp/tfidf_corpus.mm', tfidf_corpus)
    else:
        print('TFIDF corpus already created, loading now...')
        tfidf_corpus = corpora.MmCorpus('./tmp/tfidf_corpus.mm')

    return dictionary, bow_corpus, tfidf_corpus

def train_lsi(corpus, dictionary, num_topics, corpus_type):
    """
    Train the LSI model given the dataset for a given amount of topics.
    """
    #train model and save for later use
    model_filename = 'lsi_' + str(corpus_type) + '_num_topics=' + str(num_topics) + '.model'
    model_path = './tmp/' + model_filename

    if not os.path.exists(model_path):
        print(('Starting training {} lsi for num_topics = {}').format(corpus_type, num_topics))
        lsi = LsiModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, onepass=False)
        lsi.save(model_path)

    else:
        print(('{} Lsi for num_topics = {} is already created, loading now...').format(corpus_type, num_topics))
        lsi = LsiModel.load(model_path)

    #construct BOW index for trained lsi model, save for later use
    index_filename = 'index_' + str(corpus_type) + '_num_topics=' + str(num_topics) + '.mm.index'
    index_path = './tmp/' + index_filename

    if not os.path.exists(index_path):
        print(('Starting construction {} index for num_topics = {}').format(corpus_type, num_topics))
        index = similarities.MatrixSimilarity(lsi[corpus])
        index.save(index_path)
    else:
        print(('index for {} corpus with num_topics = {} is already created, loading now...').format(corpus_type, num_topics))
        index = similarities.MatrixSimilarity.load(index_path)

    return lsi, index

def query_similarity(query, dictionary, model, index, doc_ids):
    """
    Return the ranking of relevant docs given a query.
    """
    query = ra.process_text(query)
    vec_bow = dictionary.doc2bow(query)
    vec_lsi = model[vec_bow]
    sims = index[vec_lsi]
    scores = {}
    for i, score in enumerate(sims):
        score = score.item()
        doc_id = doc_ids[i]
        scores[doc_id] = score

    ranking = dict(sorted(scores.items(), key=operator.itemgetter(1), reverse=True))
    return ranking

def compute_metrics(dictionary, model, index, corpus_type, num_topics, doc_ids):
    """
    Compute MAP and nDCG scores and save to json file.
    """
    metric_path = ("./LSI_results/LSI_{}_and_{}_topics.json".format(corpus_type, num_topics))
    #check whether metrics for corpus type and num_topics were already generated
    if not os.path.exists(metric_path):

        # Get ranking of document for every query and compute the MAP and NDCG score.
        qrels, queries = ra.read_qrels()
        overall_ser = {} #ranking per query
        for qid in tqdm(qrels):
            query = queries[qid]
            ranking = query_similarity(query, dictionary, model, index, doc_ids)
            overall_ser[qid] = ranking

        # Compute model evaluation scores per query
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})
        metrics = evaluator.evaluate(overall_ser)

        with open("./LSI_results/LSI_{}_and_{}_topics.json".format(corpus_type, num_topics), "w") as writer:
            json.dump(metrics, writer, indent=1)
    else:
        print('metrics for LSI_{} with {} topics were already computed'.format(corpus_type, num_topics))

def main():
    # id2word, word2id, = load_data()
    docs_by_id = get_processed_docs()
    doc_ids = list(docs_by_id)
    documents = []
    for key in docs_by_id.keys():
        doc = docs_by_id[key]
        documents.append(doc)

    #construct dictionary and corpus
    dictionary, bow_corpus, tfidf_corpus = create_corpus_and_dict(documents)

    # train lsi, generate index
    #TRAINING LOOP
    topic_args = [10, 50, 100, 500, 1000, 2000, 5000, 10000]
    for topic_num in topic_args:
        print(('starting training tfidf lsi and index with topic_num {}').format(topic_num))
        lsi, index = train_lsi(tfidf_corpus, dictionary, num_topics=topic_num, corpus_type='tfidf')
        print(('finished training tfidf lsi and index with topic_num {}').format(topic_num))
        print(('starting training bow lsi and index with topic_num {}').format(topic_num))
        lsi, index = train_lsi(bow_corpus, dictionary, num_topics=topic_num, corpus_type='bow')
        print(('finished training bow lsi and index with topic_num {}').format(topic_num))

    #METRICS LOOP
    # COMPUTE METRICS
    topic_args = [10, 50, 100, 500, 1000, 2000]
    for topic_num in topic_args:
        # retrieve model and index for TFIDF, compute and store metrics
        lsi, index = train_lsi(tfidf_corpus, dictionary, num_topics=topic_num, corpus_type='tfidf')
        compute_metrics(dictionary=dictionary, model=lsi, index=index, corpus_type='tfidf', num_topics=topic_num, doc_ids=doc_ids)
    
        # retrieve model and index for BOW, compute and store metrics
        lsi, index = train_lsi(bow_corpus, dictionary, num_topics=topic_num, corpus_type='bow')
        compute_metrics(dictionary=dictionary, model=lsi, index=index, corpus_type='bow', num_topics=topic_num, doc_ids=doc_ids)

    #TOPICS
    tfidf_lsi, index = train_lsi(tfidf_corpus, dictionary, num_topics=500, corpus_type='tfidf')
    print("top 5 TFIDF topics")
    pprint(tfidf_lsi.print_topics(num_topics=50))
    bow_lsi, index = train_lsi(bow_corpus, dictionary, num_topics=500, corpus_type='bow')
    print("top 5 BOW topics")
    pprint(bow_lsi.print_topics(num_topics=50))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--topics', type=int, default=500,
                        help="number of topics")
    config = parser.parse_args()
    main()
