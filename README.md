# Collection search engine

Simple search engine over the Stanford Information Retrieval course collection.
Access the collection: http://web.stanford.edu/class/cs276/pa/pa1-data.zip

Authors:
- Corentin Carteau
- Alexandre Florez de la Colina
- Cédric Talbot

This project was realised as part of CentraleSupélec Information Retrieval course taught by Céline Hudelot.

## Setup instructions

### Common setup instructions

We provide a quick start setup and a complete setup. 
The following steps need to be run for both.

- Check that the python version is 3.7.6 or above.
- Install dependencies: ```pip3 install -r requirements.txt```
- Download necessary NLTK packages. In a python3 shell, enter the following:

```[python]
>>> import nltk
>>> nltk.download('stopwords')
>>> nltk.download('tagsets')
>>> nltk.download("wordnet")
>>> nltk.download('averaged_perceptron_tagger')
```

- Download the original collection at http://web.stanford.edu/class/cs276/pa/pa1-data.zip
- Unzip it and place it at the project root.

### Quick start setup

- Download the collection inverted index: https://drive.google.com/file/d/1KPwEjVXzrvLOf7a2WblP9keBtxtLdj6q/view?usp=sharing
- Unzip it and place it at the project root.
- Run the application from the project root: ```python3 app.py```
    If the application does not run, check that you have no other process running on port 5000.
    The application will take ~30 seconds to start, the time to load the inverted index in memory.
- Once the application has started, go to http://localhost:5000/ in a browser to make some queries!
- NOTE: the first query is usually ~10 times slower than the following queries.

By default, the engine returns the top 20 matches. This can be changed in the config.py file.

### Complete setup (with local inverted index computation)

- In config.py, change `compute_inverted_index` to `True`
- Run the application from the project root: ```python3 app.py```
- This will take ~40 minutes to process the collection and build the inverted index. 
- If the application does not run after, check that you have no other process running on port 5000.
- Once the application has started, go to http://localhost:5000/ in a browser to make some queries!
- In config.py, change `compute_inverted_index` back to `False` now that the inverted index is saved locally.


## Engine components

### Collection pre-processing

The collection is already tokenized. 
As additional pre-processing, all the words with less than 2 characters and english stopwords (using nltk list) are removed.
Then, the words are POS-tagged using nltk default POS-tagger and lemmatized using nltk WordNetLemmatizer.
This pre-processing is costly compared to simple stemming, but it leads to a better quality of pre-processed documents.

The preprocessed documents are saved in a separate folder with the same structure as the original collection for easier re-use.
This pre-processing step takes around 30 minutes on a standard laptop.

### Collection indexing

We use sklearn TfidfVectorizer to compute both the collection vocabulary and word count and the TF-IDF representation of each document.
The sparse matrix returned by the vectorizer is then used to build the inverted index for the collection.
This inverted index is a dictionary which key set is the vocabulary 
and values are the list of documents sorted by relevance for the corresponding term.
The relevance of a document for a term corresponds to the TF-IDF weight of this term in the vectorised document.

The inverted index is saved using pickle to be easily reloaded and re-used.
The inverted index computation and saving takes less than 10 minutes on a standard laptop.

### Document querying

- The user query is preprocessed:
    - Stop words are removed
    - Words are lemmatized with the same procedure as for documents
    - Duplicated tokens are removed
- For each word in the query, relevant documents and their TF-IDF weight for the word are retrieved from the inverted index.
- Documents for each words are combined and sorted by relevance. 

The procedure is the following:
- Compute the intersection of the retrieved documents with their weight for each term.
    Union is used if there is not enough documents in the intersection.
- Score each document with the harmonic mean of its TF-IDF weights for the different query terms.
    When the document does not contain a query term, the default weight used is half the minimum weight
    of all the retrieved documents.
- Return the <nresults> top documents (i.e. with largest harmonic mean score).

### Corpus statistics

The original corpus contains 25 510 231 terms and 354 396 distinct terms.
The pre-processed corpus contains 16 184 888 terms and 291 323 distinct terms.