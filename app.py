import pandas as pd
from flask import Flask, render_template, request

# libraries for making count matrix and similarity matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

# Read CSV File
df = pd.read_csv("movie_dataset.csv")

# Make all dataset in LowerCase
for col in df.columns:
    try:
        df[col] = df[col].apply(lambda x: x.lower())
    except:
        pass

# Select Features
features = ['keywords', 'cast', 'genres', 'director']

# Create a column in DF which combines all selected featu
for feature in features:
    df[feature] = df[feature].fillna('')


def combine_features(row):
    try:
        return row['keywords'] + " " + row['cast'] + " " + row["genres"] + " " + row["director"]
    except:
        print("Error:", row)


df["combined_features"] = df.apply(combine_features, axis=1)

# Helper Functions


def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]


def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]

# Compute the Cosine Similarity based on the count_matrix¶


def create_similarities(data_frame):
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data_frame["combined_features"])

    # creating a similarity score matrix
    cosine_sim = cosine_similarity(count_matrix)

    return cosine_sim


similarity_scores = create_similarities(df)

# Get the recommeded Movies


def get_recommendations(cosine_sim, movie_user_likes, count_recommendations):
    recommended_movies = []
    movie_user_likes = movie_user_likes.lower()

    try:
        # Compute the Cosine Similarity based on the count_matrix
        movie_index = get_index_from_title(movie_user_likes)
        similar_movies = list(enumerate(cosine_sim[movie_index]))

        # Get a list of similar movies in descending order of similarity score.
        sorted_similar_movies = sorted(
            similar_movies, key=lambda x: x[1], reverse=True)

        for i, element in enumerate(sorted_similar_movies):
            recommended_movies.append(get_title_from_index(element[0]))
            if i >= count_recommendations:
                break
    except:
        pass

    return recommended_movies


# FLASK and Deployment
app = Flask(__name__)


@app.route("/")
def home():
    return render_template('home.html')


@app.route("/recommend")
def recommend():
    movie = request.args.get('movie')
    result = get_recommendations(similarity_scores, movie, 20)

    movie = movie.upper()
    if len(result) == 0:
        return render_template('recommend.html', movie=movie, r=result, t='s')
    else:
        return render_template('recommend.html', movie=movie, r=result, t='l')


if __name__ == '__main__':
    app.run()
