import calendar
from fastapi import FastAPI, HTTPException, Query
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

# Load data
df = pd.read_csv('merged_data.csv')

# Data preprocessing
# df.loc[df['release_year'] == 1970, 'release_year'] = 0
df = df[df['release_year'] != 1970]
df = df.dropna()

# Platform grouping
platform_groups = {
    "PC": ['PC', 'Macintosh', 'Linux', 'SteamOS', 'Windows Surface'],
    "Nintendo": ['Nintendo Switch', 'Super NES', 'Wii', 'Nintendo 64', 'Wii U', 'Nintendo 3DS', 'Nintendo DS', 'NES', 'Nintendo DSi', '3DS', 'New Nintendo 3DS', 'Nintendo 64DD', 'GameCube'],
    "PlayStation": ['PlayStation 5', 'PlayStation', 'PlayStation 2', 'PlayStation 3', 'PlayStation 4', 'PlayStation Portable', 'PlayStation Vita'],
    "Xbox": ['Xbox Series X', 'Xbox', 'Xbox 360', 'Xbox One'],
    "Mobile": ['iPhone', 'iPad', 'Android', 'Windows Phone', 'iPod', 'Pocket PC', 'iOS (iPhone/iPad)'],
    "Sega": ['Genesis', 'Sega CD', 'Saturn', 'Dreamcast', 'Master System', 'Sega 32X']
}

genre_mapping = {
    'Role-playing (RPG)': 'RPG',
    'Platformer' : 'Platform',
    'Sports' : 'Sport',
    'Simulator' : 'Simulation',
    'Card & Board Game' : 'Card & Board',
    'Fightings' : 'Fighting',
}

def get_recommendations(game_title, df, similarity_matrix, n=5):
    idx = df[df['title'] == game_title].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    game_indices = [i[0] for i in sim_scores]
    recommendations = df.iloc[game_indices][['title', 'platform', 'genre', 'score']]
    logger.debug(f"Game indices: {game_indices}")
    logger.debug(f"Recommendations DataFrame: {recommendations}")
    return recommendations
        
        

@app.get("/")
def read_root():
    return {"message": "Welcome to the Game Review API"}

@app.get("/reviews-by-year")
def get_reviews_by_year():
    released = df.groupby(by='release_year')
    released = released.title.count().reset_index()
    return released.to_dict(orient='records')

@app.get("/reviews-by-month")
def get_reviews_by_month():
    df_filter = df[df['release_year'] >= 2014]
    monthly_releases = df_filter.groupby(['release_month']).size().to_dict()
    result = [(month, count) for month, count in monthly_releases.items()]
    return result

@app.get("/genre-popularity")
def get_genre_popularity():
    genres = df['genre'].str.split(', ', expand=True).stack().reset_index(level=1, drop=True)
    df_expanded = df.drop(columns=['genre']).join(genres.rename('genre'))
    df_expanded['genre'] = df_expanded['genre'].replace(genre_mapping)

    genre_stats = df_expanded.groupby('genre').agg(
        average_score=('score', 'mean'),
        num_games=('title', 'count')
    ).reset_index()

    top_15_genres = genre_stats.sort_values(by='num_games', ascending=False).head(15)

    return top_15_genres.to_dict(orient='records')

@app.get("/platform-distribution")
def get_platform_distribution():
    all_platforms = [platform for platforms in platform_groups.values() for platform in platforms]
    platform = {}
    for group, platforms in platform_groups.items():
        platform[group] = df[df['platform'].isin(platforms)].groupby(by='release_year').size()
    
    platform['Other'] = df[~df['platform'].isin(all_platforms)].groupby(by='release_year').size()
    df_platform = pd.DataFrame(platform).fillna(0).astype(int)
    df_platform = df_platform.reset_index()
    return df_platform.to_dict(orient='records')
    
@app.get("/recommend")
def recommend(game_title: str = Query(..., description="Game title"), 
              n: int = Query(default=5, description="Number of recommendations")):
    if game_title not in df['title'].values:
        raise HTTPException(status_code=404, detail="Game title not found")

    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(df['genre'].str.split(', '))
    genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_, index=df.index)

    platform_encoder = OneHotEncoder(sparse_output=False)
    platform_matrix = platform_encoder.fit_transform(df[['platform']])
    platform_df = pd.DataFrame(platform_matrix, 
                    columns=platform_encoder.get_feature_names_out(['platform']), 
                    index=df.index)

    features_df = pd.concat([genre_df, platform_df, df[['release_year', 'score']]], axis=1)
    similarity_matrix = cosine_similarity(features_df)
    logger.debug(f"Similarity matrix: {similarity_matrix}")
    
    recommendations = get_recommendations(game_title, df, similarity_matrix, n)
    logger.debug(f"Recommendations: {recommendations}")
    return recommendations.to_dict(orient='records')
