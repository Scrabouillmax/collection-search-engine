from flask import Flask, request, abort, jsonify
from indexation import preprocess_collection, build_inverted_index, read_all_documents, load_inverted_index
from config import config
from query import preprocess_query, match


app = Flask(__name__, static_url_path='')


def load():
    global inverted_index
    if config["compute_inverted_index"]:
        preprocess_collection()
        build_inverted_index(read_all_documents())
    inverted_index = load_inverted_index()


@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route('/query')
def query():
    if "q" not in request.args:
        abort(400)
    q = request.args.get('q')
    tokens = preprocess_query(q)
    documents = match(inverted_index, tokens, nresults=config["n_results"])
    top_match = ""
    if len(documents) > 0:
        f = open("{}{}".format(config["original_data"], documents[0][0]), "r")
        top_match = f.read()
        f.close()

    results = {
        "documents": documents,
        "top_match": top_match,
    }
    return jsonify(results)


if __name__ == "__main__":
    load()
    app.run()
