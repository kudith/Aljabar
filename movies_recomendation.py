import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk
import logging

nltk.download('stopwords')
nltk.download('wordnet')

# Set page configuration
st.set_page_config(layout="wide", page_title="Movie Recommendation")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
file_handler = logging.FileHandler('recommendation_log.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Function to log recommendations
def log_recommendations(title, recommendations, recommendation_type):
    logger.info(f"Recommendation Type: {recommendation_type}")
    logger.info(f"Movie: {title}")
    for idx, movie in recommendations.iterrows():
        logger.info(f"Recommended Movie: {movie['title']}, Score: {movie['imdb_rating']}")

# Load dataset
@st.cache_data
def load_data(sample_size=15000):
    movies = pd.read_csv('data/TMDB_all_movies.csv')
    movies = movies[movies['vote_count'] >= 50]
    movies.fillna('', inplace=True)

    # Explicitly cast columns to string where necessary
    string_columns = ['overview', 'genres', 'tagline', 'cast', 'director']
    for col in string_columns:
        movies[col] = movies[col].astype(str)

    # Convert 'imdb_rating' column to numeric, forcing errors to NaN
    movies['imdb_rating'] = pd.to_numeric(movies['imdb_rating'], errors='coerce')

    # Drop rows with NaN values in 'imdb_rating'
    movies = movies.dropna(subset=['imdb_rating'])

    # Sample the dataset
    movies = movies.sample(sample_size, random_state=42).reset_index(drop=True)

    # Combine text features
    movies['combined_features'] = movies['overview'] + ' ' + movies['genres'] + ' ' + movies['tagline'] + ' ' + movies['cast'] + ' ' + movies['director']
    return movies

with st.spinner('Loading data...'):
    movies = load_data()

# Preprocess text data
@st.cache_data
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

movies['processed_features'] = movies['combined_features'].apply(preprocess_text)

# Calculate TF-IDF matrix
@st.cache_data
def calculate_tfidf_matrix(movies):
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    return tfidf.fit_transform(movies['processed_features'])

tfidf_matrix = calculate_tfidf_matrix(movies)

# Calculate cosine similarity matrix
@st.cache_data
def calculate_cosine_similarity(_tfidf_matrix):
    return linear_kernel(_tfidf_matrix, _tfidf_matrix)

cosine_sim = calculate_cosine_similarity(tfidf_matrix)

# Train KNN model
@st.cache_data
def train_knn_model(_tfidf_matrix):
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(_tfidf_matrix)
    return knn

knn = train_knn_model(tfidf_matrix)

# Function to recommend movies based on genre
def get_genre_recommendations(genre, min_rating=6.5):
    genre_movies = movies[movies['genres'].str.contains(genre, case=False)]
    return genre_movies[genre_movies['imdb_rating'] > min_rating]

# Function to recommend movies
def get_recommendations(title, knn_model=knn, cosine_sim=cosine_sim, min_rating=6.5):
    idx = movies[movies['title'] == title].index[0]
    distances, indices = knn_model.kneighbors(tfidf_matrix[idx], n_neighbors=21)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    movie_indices = [i[0] for i in sim_scores]
    recommendations = movies.iloc[movie_indices]
    recommendations = recommendations[recommendations['imdb_rating'] > min_rating]
    recommendations = recommendations.sort_values(by='imdb_rating', ascending=False).head(20)
    log_recommendations(title, recommendations, "Content-Based")
    return recommendations

# Function to recommend movies using collaborative filtering
def get_collaborative_recommendations(title, min_rating=6.5):
    recommendations = get_recommendations(title, min_rating=min_rating)
    log_recommendations(title, recommendations, "Collaborative Filtering")
    return recommendations.sort_values(by='imdb_rating', ascending=False).head(20)

# Function to recommend movies using hybrid approach
def get_hybrid_recommendations(title, genre, min_rating=6.5):
    content_recommendations = get_recommendations(title, min_rating=min_rating)
    genre_recommendations = get_genre_recommendations(genre, min_rating=min_rating)
    hybrid_recommendations = pd.concat([content_recommendations, genre_recommendations]).drop_duplicates()
    hybrid_recommendations = hybrid_recommendations.sort_values(by='imdb_rating', ascending=False).head(20)
    log_recommendations(title, hybrid_recommendations, "Hybrid")
    return hybrid_recommendations

# Function to visualize recommendations
def visualize_recommendations(recommendations):
    fig, ax = plt.subplots(figsize=(10, 6))
    ratings = recommendations['imdb_rating'].values
    titles = recommendations['title'].values
    y_pos = np.arange(len(titles))

    ax.barh(y_pos, ratings, align='center', color='skyblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(titles)
    ax.invert_yaxis()
    ax.set_xlabel('Rating')
    ax.set_title('Recommended Movies')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

    # Add ratings on the bars
    for i in range(len(ratings)):
        ax.text(ratings[i] + 0.1, y_pos[i], f'{ratings[i]:.1f}', va='center', color='black')

    st.pyplot(fig)

# Function to visualize TF-IDF vectors
def visualize_tfidf_vectors(tfidf_matrix, movie_indices):
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(tfidf_matrix[movie_indices].toarray())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], color='skyblue')

    for i, idx in enumerate(movie_indices):
        ax.annotate(movies.iloc[idx]['title'], (reduced_vectors[i, 0], reduced_vectors[i, 1]))

    ax.set_title('TF-IDF Vectors of Recommended Movies')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

    st.pyplot(fig)

# Function to visualize KNN using Seaborn
def visualize_knn(tfidf_matrix, movie_index, knn_model=knn):
    distances, indices = knn_model.kneighbors(tfidf_matrix[movie_index], n_neighbors=21)
    indices = indices.flatten()

    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(tfidf_matrix[indices].toarray())

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=reduced_vectors[:, 0], y=reduced_vectors[:, 1], ax=ax, s=100, color='skyblue', edgecolor='w', linewidth=1.5)

    for i, idx in enumerate(indices):
        if idx == movie_index:
            ax.annotate(movies.iloc[idx]['title'], (reduced_vectors[i, 0], reduced_vectors[i, 1]), color='red', weight='bold')
        else:
            ax.annotate(movies.iloc[idx]['title'], (reduced_vectors[i, 0], reduced_vectors[i, 1]))

    ax.set_title('KNN Visualization of Selected Movie')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

    st.pyplot(fig)

# Streamlit UI
st.markdown("""
    <style>
    body {
        background-color: #141414;
        color: white;
    }
    .movie-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin: 10px;
    }
    .movie-title {
        font-size: 14px;
        font-weight: bold;
        margin-top: 5px;
        text-align: center;
    }
    .movie-info {
        font-size: 12px;
        text-align: center;
        color: #B3B3B3;
    }
    .movie-poster {
        width: 150px;
        height: 225px;
        border-radius: 5px;
    }
    .movie-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎬 Movie Recommendation System")

# Search and recommendation section
movie_name = st.text_input("🔎 Enter a movie name to get recommendations:", "")
recommendation_type = st.selectbox("Select recommendation type:", ["Content-Based", "Collaborative Filtering", "Hybrid"])

if movie_name:
    with st.spinner('Finding movies...'):
        # Use fuzzy matching to find the closest match
        movie_titles = movies['title'].tolist()
        closest_matches = process.extract(movie_name, movie_titles, limit=1)
        matching_movies = movies[movies['title'].isin([match[0] for match in closest_matches])]

        if not matching_movies.empty:
            selected_movie = matching_movies.iloc[0]
            st.write(f"### Selected Movie: **{selected_movie['title']}**")
            if selected_movie['poster_path']:
                st.image(f"https://image.tmdb.org/t/p/w500{selected_movie['poster_path']}", width=200)
            else:
                st.write("Poster not available.")

            # Display additional details
            st.write(f"**Genres:** {selected_movie['genres']}")
            st.write(f"**Rating:** {selected_movie['imdb_rating']}")
            st.write(f"**Cast:** {selected_movie['cast']}")
            st.write(f"**Director:** {selected_movie['director']}")
            st.write(f"**Overview:** {selected_movie['overview']}")
            st.write(f"**Tagline:** {selected_movie['tagline']}")

            # Get recommendations based on selected type
            if recommendation_type == "Content-Based":
                recommendations = get_recommendations(selected_movie['title'])
            elif recommendation_type == "Collaborative Filtering":
                recommendations = get_collaborative_recommendations(selected_movie['title'])
            elif recommendation_type == "Hybrid":
                recommendations = get_hybrid_recommendations(selected_movie['title'], selected_movie['genres'])

            st.write("### Recommended Movies:")
            cols = st.columns(5)
            for i, movie in enumerate(recommendations.iterrows()):
                with cols[i % 5]:
                    genres = movie[1]['genres'].strip('[]').replace("'", "").split(',')
                    st.markdown(f"""
                        <div class="movie-container">
                            <img src="https://image.tmdb.org/t/p/w500{movie[1]['poster_path']}" class="movie-poster"/>
                            <div class="movie-title">{movie[1]['title']}</div>
                            <div class="movie-info">Genres: {', '.join(genres)} | Rating: {movie[1]['imdb_rating']}</div>
                        </div>
                    """, unsafe_allow_html=True)

            # Visualize recommendations
            visualize_recommendations(recommendations)

            # Visualize TF-IDF vectors
            movie_indices = recommendations.index.tolist()
            visualize_tfidf_vectors(tfidf_matrix, movie_indices)

            # Visualize KNN
            visualize_knn(tfidf_matrix, selected_movie.name)
        else:
            st.write("No movies found with the given name.")

# Popular movies section
st.write("### 🌟 Popular Movies:")
with st.spinner('Loading popular movies...'):
    # Convert 'imdb_rating' column to numeric, forcing errors to NaN
    movies['imdb_rating'] = pd.to_numeric(movies['imdb_rating'], errors='coerce')

    # Filter movies with rating above 6.5
    popular_movies = movies[movies['imdb_rating'] > 6.5]
    # Sort by popularity and rating
    popular_movies = popular_movies.sort_values(by=['popularity', 'imdb_rating'], ascending=[False, False]).head(50)
    cols = st.columns(5)
    for i, movie in enumerate(popular_movies.iterrows()):
        with cols[i % 5]:
            genres = movie[1]['genres'].strip('[]').replace("'", "").split(',')
            st.markdown(f"""
                <div class="movie-container">
                    <img src="https://image.tmdb.org/t/p/w500{movie[1]['poster_path']}" class="movie-poster"/>
                    <div class="movie-title">{movie[1]['title']}</div>
                    <div class="movie-info">Genres: {', '.join(genres)} | Rating: {movie[1]['imdb_rating']}</div>
                </div>
            """, unsafe_allow_html=True)