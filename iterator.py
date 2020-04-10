import os


def iterator(path):
    """
    Iterator over the document collection.
    :param path: Path to the root folder of the collection (i.e. "./pa1-data/")
    :return: Iterator over the filenames of the collection
        Filename format: <subfolder>/<filename> (i.e. "1/facs.stanford.edu_resources")
    """
    for root, subdirs, files in os.walk(path):
        if len(subdirs) == 0:
            for f in files:
                yield "{}/{}".format(root.replace(path, ""), f)
