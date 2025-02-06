import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import requests
import json
import urllib.request
import pickle
from flask.json import jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# Load the Sentiment Analysis Model
# Load the CountVectorizer
cv = pickle.load(open('cv.pkl', 'rb'))
sentiment_model = pickle.load(open('f44.pkl', 'rb'))
filename = 'top_dict.pkl'
clf = pickle.load(open(filename, 'rb'))
movies=pd.DataFrame(clf)
new_df = pickle.load(open('new_df.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))
    
app = Flask(__name__)
def fetch_movie_reviews(movie_id, api_key):
    # Make an API request to fetch movie reviews
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={api_key}')
    
    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        return data.get('results', [])
    else:
        # Handle the error case
        return []

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    movie_titles = new_df['title'].tolist()
    return jsonify(movie_titles)

@app.route('/id',methods=['GET'])
def id():
    movies_id = new_df['id'].tolist()
    return jsonify(movies_id)

@app.route('/')
def hello_world():
   data_from_backend={'Title':movies['title'].values,'Poster_parth':'http://image.tmdb.org/t/p/w185//'+movies['poster_path'].values,'ID':movies['id'].values}
   return render_template('Home.html',data=data_from_backend)

@app.route('/recommend')
def recommend_ui():
    
    return render_template('recommend.html')

@app.route('/recommend_movies', methods=['POST'])
def recommend():
    user_input = request.form.get('user_input')

    if not user_input:
        # Handle case where no input is provided
        return render_template('recommend.html', data=[])

    matching_rows = new_df[new_df['title'] == user_input]

    if matching_rows.empty:
        # Handle case where the entered movie title is not found
        return render_template('recommend.html', data=[])

    try:
        movie_index = matching_rows.index[0]
        distance = similarity[movie_index]

        if len(distance) == 0:
            # Handle case where the distance array is empty
            return render_template('recommend.html', data=[])

        movie_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[0:20]
        data = []
        for i in movie_list:
            item = []
            item.append(new_df.iloc[i[0]].poster_path)
            item.append(new_df.iloc[i[0]].title)
            item.append(new_df.iloc[i[0]].id)
            data.append(item)
        return render_template('recommend.html', data=data)
    except IndexError:
        # Handle other potential errors
        return render_template('recommend.html', data=[])
api_key = '34138ccc2c18882f82644558b4af73e6'

@app.route('/tr')
def tr_ui():
    movie_id = request.args.get('id')
    reviews_list = []  # list of reviews
    reviews_status = []  # list of comments (good or bad)

    # Fetch movie reviews using the TMDB API
    reviews = fetch_movie_reviews(movie_id, api_key)

    for review in reviews:
        if 'content' in review and isinstance(review['content'], str):
            review_content = review['content'].lower()
            reviews_list.append(review_content)
            movie_review_list = np.array([review_content])
            
            # Transform new data using the loaded CountVectorizer
            movie_vector = cv.transform(movie_review_list)
            
            pred = sentiment_model.predict(movie_vector)
            reviews_status.append('Good' if pred else 'Bad')

    movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}
    id_from_url = request.args.get('id')

    # Use the id to filter matching rows in your DataFrame
    matching_rows = new_df[new_df['id'] == int(id_from_url)]

    if matching_rows.empty:
        # Handle case where the entered movie id is not found
        return render_template('recommend.html', data=[])

    try:
        movie_index = matching_rows.index[0]
        distance = similarity[movie_index]

        if len(distance) == 0:
            # Handle case where the distance array is empty
            return render_template('recommend.html', data=[])

        movie_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:20]
        data = []
        for i in movie_list:
            item = []
            item.append(new_df.iloc[i[0]].poster_path)
            item.append(new_df.iloc[i[0]].title)
            item.append(new_df.iloc[i[0]].id)
            data.append(item)
        return render_template('tr.html', data=data,reviews=movie_reviews)
    except IndexError:
        # Handle other potential errors
        return render_template('tr.html', reviews=movie_reviews,data=[])
@app.route('/About')
def About():
   
   return render_template('About.html')
@app.route('/Contact')
def Contact():
   
   return render_template('Contact.html')
if __name__ == "__main__":
    app.run(debug=True)
