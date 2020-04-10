from tqdm import tqdm
from iterator import iterator
from config import config


def count_terms():
    """
    98998 documents
    Results:
    Original documents:
        Duration on laptop: 19min 03s, 86.55 iterations/s
        Terms:  25 510 231
        Distinct terms:  354 396
    Pre-processed documents:
        Duration on laptop: 13min 07s, 125.70 iterations/s
        Terms:  16 184 888
        Distinct terms:  291 323
    """
    it = iterator(config["original_data"])
    terms = 0
    distinct = set()
    for file in tqdm(it):
        f = open(file, "r")
        content = f.read().split(' ')
        terms += len(content)
        distinct = distinct | set(content)

    print("Terms: ", terms)
    print("Distinct terms: ", len(distinct))


count_terms()
