from flask import Flask
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

app = Flask(__name__)


@app.route("/")
def home():
    return 'ddddddddd'


if __name__ == '__main__':
    app.run()
