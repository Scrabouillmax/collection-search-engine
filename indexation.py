import nltk
import pickle
from time import time
from tqdm import tqdm
from pathlib import Path
from nltk.corpus import stopwords, wordnet
from sklearn.feature_extraction.text import TfidfVectorizer

from config import config
from iterator import iterator


"""
##### PRE-PROCESSING #####
"""


def penn_to_wn(penn_tag):
    """
    Source: https://stackoverflow.com/questions/25534214/nltk-wordnet-lemmatizer-shouldnt-it-lemmatize-all-inflections-of-a-word
    Convert a Penn tree-bank POS-tag to a WordNet POS-tag.
    Returns None for all tags other than: nouns, adjectives, adverbs and verbs.
    :param penn_tag: Penn tree-bank POS-tag: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    :return: None or WordNet tag: https://wordnet.princeton.edu/documentation/wndb5wn
    """
    if penn_tag.startswith('J'):
        return wordnet.ADJ
    elif penn_tag.startswith('V'):
        return wordnet.VERB
    elif penn_tag.startswith('N'):
        return wordnet.NOUN
    elif penn_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def remove_stop_words(tokens):
    """
    Remove english stop words and short words (length <= 2) from token list.
    :param tokens: token list.
    :return: filtered token list.
    """
    remove = set(stopwords.words('english'))
    return [token for token in tokens if token not in remove and len(token) > 2]


def lemmatize(tokens):
    """
    Lemmatize POS-tagged token list using nltk's WordNetLemmatizer.
    :param tokens: List of tuples (token, WordNet_tag).
    :return: List of tuples (lemmatized_token, WordNet_tag)
    """
    tagged_words = nltk.pos_tag(tokens)
    tagged_words = [(token, penn_to_wn(tag)) for token, tag in tagged_words if penn_to_wn(tag) is not None]
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(token, tag) for token, tag in tagged_words]


def preprocess_collection():
    """
    Preprocess original collection:
        - remove_stop_words
        - lemmatize
    Saves the preprocessed files under the path specified in config, keeping the original collection structure.
    Execution duration on laptop: 27min 46s, 59.40 iterations/s
    """
    it = iterator(config["original_data"])

    Path(config["preprocessed_data"]).mkdir(parents=True, exist_ok=True)
    for i in range(10):
        Path("{}{}/".format(config["preprocessed_data"], i)).mkdir(parents=True, exist_ok=True)

    print("Pre-processing collection")
    t = time()
    for file in tqdm(it):
        f = open("{}{}".format(config["original_data"], file), "r")
        words = f.read().split(' ')
        f.close()
        words = remove_stop_words(words)
        words = lemmatize(words)
        new_content = " ".join(words)
        new_file = open("{}{}".format(config["preprocessed_data"], file), "w+")
        new_file.write(new_content)
        new_file.close()
    print("Duration:", time() - t)


"""
##### Document representation #####
"""


def read_all_documents():
    """
    Read all documents from pre-processed collection.
    :return: Documents content as a string list.
    """
    documents = []
    for filename in tqdm(iterator(config["preprocessed_data"])):
        f = open("{}{}".format(config["preprocessed_data"], filename), "r")
        documents.append(f.read())
        f.close()
    return documents


def build_inverted_index(documents):
    """
    Builds and save collection inverted index.
    The index is a dictionary mapping each token to a list of documents, sorted by relevance order.
    List items format for each token: (document_index_in_collection, token_tf_idf_weight_in_document)
    :param documents: List of string.
    """
    print("Building tf-idf representation")
    t = time()
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)  # docs, words
    feature_names = vectorizer.get_feature_names()
    print("Duration:", time() - t)
    t = time()

    print("Building inverted index")
    inverted_index = {word: [] for word in feature_names}
    for row, col in zip(*vectors.nonzero()):
        weight = vectors[row, col]
        word = feature_names[col]
        inverted_index[word].append((row, weight))
    print("Duration:", time() - t)
    t = time()

    print("Sorting inverted index")
    for word in feature_names:
        inverted_index[word] = sorted(inverted_index[word], key=lambda t: t[1], reverse=True)
    print("Duration:", time() - t)
    t = time()

    print("Saving inverted index")
    with open('inverted_index.pickle', 'wb') as handle:
        pickle.dump(inverted_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Duration:", time() - t)


def load_inverted_index():
    """
    Load the collection inverted index from pickle file.
    """
    t = time()
    print("Loading inverted index")
    with open('inverted_index.pickle', 'rb') as handle:
        inverted_index = pickle.load(handle)
    print("Duration:", time() - t)
    return inverted_index
