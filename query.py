from time import time
from itertools import chain
from scipy.stats import hmean

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
    Match a query against the collection inverted index.
    Matching procedure:
    - Compute intersection of corresponding documents with their weight for each term.
        Union is used if there is not enough documents in the intersection
    - Score each document with the harmonic mean of its weights for the different terms.
        When the document does not contain a query term, the default weight used is half the minimum weight
        of all the matched documents.
    - Return the <nresults> top documents (i.e. with largest score).
    :param inverted_index: Inverted index representation of the collection.
    :param query_tokens: List of tokens, the pre-processed query.
    :param nresults: Maximum number of results to return
    :return: List of the <nresults> most relevant documents names with their relevance score.
    """
    print("Matching documents for query: {}".format(" ".join(query_tokens)))
    t = time()

    documents_per_token = []
    weights_per_token = []
    for token in query_tokens:
        try:
            docs_weights = inverted_index[token]
            documents_per_token.append([doc for doc, _ in docs_weights])
            weights_per_token.append([weight for _, weight in docs_weights])
        except:
            # token not found in collection inverted index.
            pass

    unique_doc_names = set.intersection(*[set(docs) for docs in documents_per_token])
    union = len(unique_doc_names) < nresults
    placeholder_weight = 100
    if union:
        unique_doc_names = set.union(*[set(docs) for docs in documents_per_token])
        for weight in chain(*weights_per_token):
            if weight < placeholder_weight:
                placeholder_weight = weight
        placeholder_weight /= 2

    weights_per_doc = {name: [] for name in unique_doc_names}
    for name, weight in zip(chain(*documents_per_token), chain(*weights_per_token)):
        if union or name in unique_doc_names:
            weights_per_doc[name].append(weight)

    results = []
    for name, weights in weights_per_doc.items():
        if union and len(weights) < len(documents_per_token):
            weights += [placeholder_weight] * (len(documents_per_token) - len(weights))
        results.append((name, hmean(weights)))

    results = sorted(results, key=lambda res: res[1], reverse=True)[:nresults]

    index_to_name = list(iterator(config["original_data"]))
    results = [(index_to_name[i], w) for i, w in results]

    print("Found {} documents in {}s".format(len(results), time() - t))
    return results
