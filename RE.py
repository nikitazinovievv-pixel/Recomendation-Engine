import os
import sys

# Suppress native stderr output (C++ logs, PyTorch redirect notes) during imports
sys.stderr.flush()
old_stderr_fd = os.dup(2)
devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, 2)
os.close(devnull)

try:
    import torch
    from sentence_transformers import SentenceTransformer
finally:
    # Restore normal stderr output for your script
    os.dup2(old_stderr_fd, 2)
    os.close(old_stderr_fd)


import pandas as pd
import numpy as np 
import pathlib as pl
from  sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


movie_title = input()

N_path = pl.Path('/Users/nikitazinovev/Desktop/Study/Semester 3:4/Tools/Projects/netflix_titles.csv')
N_data = pd.read_csv(N_path)
N_data = N_data.dropna(ignore_index=True)

del N_data['date_added']
# Replace full country names to compact ones

country_map = {
    'United States': 'USA',
    'United Kingdom': 'UK',
    'United Arab Emirates': 'UAE',
    'South Africa': 'SA',
    'South Korea': 'SK',
    'Hong Kong': 'HK',
    'New Zealand': 'NZ',
    'Czech Republic': 'CR'
}
N_data = N_data.replace(country_map, regex=True)


if movie_title not in N_data['title'].values:
    print("No such Movie")

else:
    cols_to_use = ['type','title', 'director', 'cast', 'country', 'rating', 'listed_in', 'description']
    N_data[cols_to_use] = N_data[cols_to_use].fillna('')

    N_data['combined_text'] = (
        "Type: " + N_data['type'].astype(str) +
        N_data['title'].astype(str) +
        ". Directed by: " + N_data['director'].astype(str) +
        ". Starring: " + N_data['cast'].astype(str) +
        ". Country: " + N_data['country'].astype(str) +
        ". Rating: " + N_data['rating'].astype(str) +
        ". Genres: " + N_data['listed_in'].astype(str) +
        ". Overview: " + N_data['description'].astype(str)
    )



    model = SentenceTransformer('all-MiniLM-L6-v2')

    description_embeddings = model.encode(
        N_data['combined_text'].tolist(), 
        show_progress_bar=True
    )



    #movie_title = "Jaws"
    target_idx = N_data[N_data['title'].str.lower() == movie_title.lower()].index[0]
    print(target_idx)
    # 1. Compute pairwise similarity matrix (Shape: num_titles x num_titles)
    similarity_matrix = cosine_similarity(description_embeddings)

    # 2. Function to get top K recommendations for a given title index
    def recommend(title_idx, top_k=15):
        # Get similarity scores for the selected item against all others
        scores = similarity_matrix[title_idx]
        
        # Sort indices in descending order (excluding the item itself at index 0)
        top_indices = np.argsort(scores)[::-1][1:top_k+1]
        
        return N_data.iloc[top_indices][['title', 'listed_in']]

    recommendations = recommend(target_idx, top_k=15)
    print(f'Top 15 movies if you like "{movie_title}":\n{recommendations}')
    
    target_idx = N_data[N_data['title'].str.lower() == movie_title.lower()].index[0]
    print(target_idx)

