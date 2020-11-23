import os
import argparse
import pickle as pkl
import read_ap as ra
from pprint import pprint
from LSI import create_corpus_and_dict

from gensim.test.utils import get_tmpfile
from gensim.models import LdaModel
import gensim.matutils

from read_ap import get_processed_docs

def train_LDA(corpus, id2word, num_topics):
    """
    Train LDA model with specified number of topics.
    """
    lda_model = LdaModel(corpus=corpus,
                         id2word=id2word,
                         num_topics=num_topics
                         )
    return lda_model


def main():
    docs_by_id = get_processed_docs()
    doc_ids = list(docs_by_id)
    documents = []
    for key in docs_by_id.keys():
        doc = docs_by_id[key]
        documents.append(doc)

    # construct dictionary and corpus
    dictionary, bow_corpus, tfidf_corpus = create_corpus_and_dict(documents)

    if not os.path.exists( ("./LDA_MODELS/BOW_LDA_{}_TOPICS.model".format(config.topics)) ):
        print("Starting LDA with topics = {} with BOW training now.".format(config.topics))
        BOW_LDA = train_LDA(bow_corpus,
                              dictionary,
                              config.topics
                            )
        BOW_LDA.save( ("./LDA_MODELS/BOW_LDA_{}_TOPICS.model".format(config.topics)) )

    else:
        print("LDA with BOW already trained, loading the file.")
        BOW_LDA = LdaModel.load( ("./LDA_MODELS/BOW_LDA_{}_TOPICS.model".format(config.topics)) )

    # Print found topics
    # for i in range(config.topics):
    #     pprint(BOW_LDA.print_topic(i))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--topics', type=int, default=500,
                        help="number of topics")

    config = parser.parse_args()
    main()


