from flask import Flask, request
from spacy.cli import download
from spacy import load
import numpy as np
from pandas import DataFrame
from math import e
download("en_core_web_md")
nlp = load("en_core_web_md")

def similarity(s1,s2):
    # convert to lowercase
    s1 = s1.lower()
    s2 = s2.lower()
    
    # Convert sentences to doc
    docs = [nlp(s) for s in [s1,s2]]
    
    # Tokenize the sentence
    token1,token2 = [[token.text for token in doc] for doc in docs] 
    jac =  len(set(token1).intersection(token2)) / len(set(token1).union(token2))
    

    # Vectorize using word2vector
    v1,v2 = [doc.vector for doc in docs]
    cos = np.dot(v1,v2) / sum(v1**2)**0.5 / sum(v2**2)**0.5
    
    # Eucledian distance
    euc = e ** -sum((v1-v2)**2)**0.5
    
    df = DataFrame([[jac,cos,euc]], index = ["Similarity"], columns = ["Jaccard", "Cosine","Eucledian"]).T
    arr = np.round(np.array(df['Similarity']),3)
    return arr

app = Flask(__name__)

@app.route("/")
def index():
    return HOME_HTML
HOME_HTML = open("home.html").read()

@app.route("/run_model")
def greet():
    text1 = request.args.get('text1')
    text2 = request.args.get('text2')
    return open("model_results.html").read().format(text1,text2,*similarity(text1,text2))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
