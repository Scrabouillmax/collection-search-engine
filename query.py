from time import time

from config import config
from iterator import iterator
from indexation import remove_stop_words, lemmatize


def preprocess_query(query):
    """
    Preprocess query:
        - remove stop words
        - lemmatize
        - remove duplicate tokens
    :param query: query string
    :return: list of tokens.
    """
    tokens = query.split(' ')
    tokens = remove_stop_words(tokens)
    tokens = lemmatize(tokens)
    tokens = list(set(tokens))

    return tokens


def match(inverted_index, query_tokens, nresults=20):
    """
    TODO improve matching and write description
    :param inverted_index: Inverted index representation of the collection.
    :param query_tokens: List of tokens, the pre-processed query.
    :param nresults: Maximum number of results to return
    :return: List of the <nresults> most relevant documents names with their relevance score.
    """
    print("Matching documents for query: {}".format(" ".join(query_tokens)))
    t = time()
    matching_documents = []
    for token in query_tokens:
        try:
            matching_documents += inverted_index[token]
        except Exception:
            pass

    unique_docs = {}
    for doc, weight in matching_documents:
        if doc in unique_docs:
            unique_docs[doc] = max(unique_docs[doc], weight)
        else:
            unique_docs[doc] = weight

    unique_docs_list = unique_docs.items()
    unique_docs_list = sorted(unique_docs_list, key=lambda x: x[1], reverse=True)

    unique_docs_list = unique_docs_list[:nresults]

    index_to_name = list(iterator(config["preprocessed_data"]))

    result = [(index_to_name[i], w) for i, w in unique_docs_list]

    print("Found {} documents in {}s".format(len(result), time() - t))
    return result
