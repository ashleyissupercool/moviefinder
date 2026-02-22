from search import load_data
from vector_search import index_movies

df = load_data("data/TMDB_movies.csv")
index_movies(df, force_reindex=True)
print("Done!")
