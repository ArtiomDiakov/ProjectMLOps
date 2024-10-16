from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import List
from functools import lru_cache
from app.routes import router

app = FastAPI()

app.include_router(router)

###############################################################################################################
### User-Item Recommendations

# Pydantic models
class UserRequest(BaseModel):
    user_id: str

class RecommendationResponse(BaseModel):
    user_id: str
    recommended_games: List[str]

# Data loading functions with caching
@lru_cache(maxsize=1)
def load_df_user_game():
    df = pd.read_parquet('../data/df_user_game.parquet')
    df['game_id'] = df['game_id'].astype(str)
    return df

@lru_cache(maxsize=1)
def load_similarity_df():
    return pd.read_parquet('../data/similarity_df.parquet')

@lru_cache(maxsize=1)
def load_df_filtered_games():
    return pd.read_parquet('../data/df_filtered_games.parquet')

@lru_cache(maxsize=1)
def load_cosine_sim_games():
    return joblib.load('../data/cosine_sim_games.pkl')

@lru_cache(maxsize=1)
def load_indices_games():
    return joblib.load('../data/indices_games.pkl')

def recomendacion_contenido(game_name):
    cosine_sim_games = load_cosine_sim_games()
    indices_games = load_indices_games()
    df_filtered_games = load_df_filtered_games()

    if game_name not in indices_games:
        return []

    idx = indices_games[game_name]
    sim_scores = list(enumerate(cosine_sim_games[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    game_indices = [i[0] for i in sim_scores]
    return df_filtered_games['app_name'].iloc[game_indices].tolist()

def recomendacion_usuario(user_id, num_recommendations=5):
    df_user_game = load_df_user_game()
    similarity_df = load_similarity_df()
    df_filtered_games = load_df_filtered_games()

    if user_id in similarity_df.index:
        similar_users = similarity_df[user_id].sort_values(ascending=False)
        similar_users = similar_users.drop(labels=[user_id])
        top_similar_users = similar_users.head(5).index
        recommended_games = df_user_game[df_user_game['user_id'].isin(top_similar_users)]['game_id'].tolist()
        user_games = df_user_game[df_user_game['user_id'] == user_id]['game_id'].tolist()
        recommended_games = [game for game in recommended_games if game not in user_games]
        recommended_games_series = pd.Series(recommended_games).value_counts()
        recommended_games = recommended_games_series.head(num_recommendations).index.tolist()
        recommended_game_names = df_filtered_games[df_filtered_games['id'].astype(str).isin(recommended_games)]['app_name'].unique().tolist()
        return recommended_game_names
    else:
        user_games = df_user_game[df_user_game['user_id'] == user_id]['game_id'].tolist()
        if not user_games:
            return ["user_not_found"]
        user_game_names = df_filtered_games[df_filtered_games['id'].astype(str).isin(user_games)]['app_name'].tolist()
        if not user_game_names:
            return []
        recommended_game_names = []
        for game in user_game_names:
            recs = recomendacion_contenido(game)
            recommended_game_names.extend(recs)
        recommended_game_names = [game for game in recommended_game_names if game not in user_game_names]
        recommended_game_names = list(pd.Series(recommended_game_names).drop_duplicates().head(num_recommendations))
        return recommended_game_names

@app.post("/recommend", response_model=RecommendationResponse)
def recommend_games(user_request: UserRequest):
    user_id = user_request.user_id
    recommendations = recomendacion_usuario(user_id)
    if recommendations is None or not isinstance(recommendations, list):
        raise HTTPException(status_code=404, detail=f"No recommendations could be generated for user {user_id}.")
    if not recommendations:
        return RecommendationResponse(user_id=user_id, recommended_games=[])
    return RecommendationResponse(user_id=user_id, recommended_games=recommendations)

###################################################################################################################################
# Developer Stats Endpoint

class DeveloperStats(BaseModel):
    year: int
    cantidad_items: int
    contenido_free: float

class DeveloperResponse(BaseModel):
    developer: str
    stats: List[DeveloperStats]

@lru_cache(maxsize=1)
def load_fixed_steam_games():
    df = pd.read_parquet('../data/fixed_steam_games.parquet')
    df['price'] = df['price'].astype(float)
    df['free'] = df['price'] == 0
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['year'] = df['release_date'].dt.year
    return df

def developer(desarrollador: str):
    df_filtered_games = load_fixed_steam_games()
    df_dev = df_filtered_games[df_filtered_games['developer'].str.contains(desarrollador, case=False, na=False)].copy()
    if df_dev.empty:
        return None
    df_summary = df_dev.groupby('year').agg(
        cantidad_items=('id', 'count'),
        contenido_free=('free', lambda x: round((x.sum() / x.count()) * 100, 2))
    ).reset_index()
    df_summary = df_summary.sort_values(by='year', ascending=False)
    result = df_summary.to_dict(orient='records')
    return result

@app.get("/developer", response_model=DeveloperResponse)
def get_developer_stats(desarrollador: str):
    stats = developer(desarrollador)
    if stats is None or len(stats) == 0:
        raise HTTPException(status_code=404, detail=f"No data found for developer: {desarrollador}")
    return DeveloperResponse(developer=desarrollador, stats=stats)

#########################################################################################################################################################
# User Data Endpoint

class UserDataResponse(BaseModel):
    usuario: str
    dinero_gastado: str
    porcentaje_recomendacion: str
    cantidad_items: int

@lru_cache(maxsize=1)
def load_df_users_items():
    return pd.read_parquet('../data/fixed_users_items.parquet')

@lru_cache(maxsize=1)
def load_df_steam_games():
    df = pd.read_parquet('../data/fixed_steam_games.parquet')
    df['price'] = df['price'].fillna(0).astype(float)
    return df

@lru_cache(maxsize=1)
def load_df_user_reviews():
    return pd.read_parquet('../data/user_reviews_with_sentiment.parquet')

def userdata(user_id: str):
    df_users_items = load_df_users_items()
    df_steam_games = load_df_steam_games()
    df_user_reviews = load_df_user_reviews()

    user_data = df_users_items[df_users_items['user_id'] == user_id]
    if user_data.empty:
        return None  # User not found

    cantidad_items = user_data.iloc[0]['items_count']
    user_items = user_data.iloc[0]['items']
    dinero_gastado_str = "0.00 USD"

    # Updated condition to check if user_items is not None and has elements
    if user_items is not None and len(user_items) > 0:
        df_user_items = pd.json_normalize(user_items)
        item_id_column = None

        # Detect the correct column for item IDs
        if 'item_id' in df_user_items.columns:
            item_id_column = 'item_id'
        elif 'appid' in df_user_items.columns:
            item_id_column = 'appid'

        if item_id_column:
            df_user_items[item_id_column] = df_user_items[item_id_column].astype(int)
            df_user_items = df_user_items.merge(
                df_steam_games[['id', 'price']], left_on=item_id_column, right_on='id', how='left'
            )
            df_user_items['price'] = df_user_items['price'].fillna(0)
            dinero_gastado = df_user_items['price'].sum()
            dinero_gastado_str = f"{dinero_gastado:.2f} USD"
    else:
        # Handle the case where user_items is None or empty
        cantidad_items = 0  # Assuming no items
        dinero_gastado_str = "0.00 USD"

    # Get the user's reviews
    user_reviews = df_user_reviews[df_user_reviews['user_id'] == user_id]

    if not user_reviews.empty:
        porcentaje_recomendacion = (user_reviews['recommend'].sum() / user_reviews.shape[0]) * 100
        porcentaje_recomendacion_str = f"{porcentaje_recomendacion:.2f}%"
    else:
        porcentaje_recomendacion_str = "No reviews available"

    # Prepare the response
    response = {
        "usuario": user_id,
        "dinero_gastado": dinero_gastado_str,
        "porcentaje_recomendacion": porcentaje_recomendacion_str,
        "cantidad_items": cantidad_items
    }

    return response

# Endpoint
@app.get("/userdata", response_model=UserDataResponse)
def get_user_data(user_id: str):
    result = userdata(user_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found.")
    return UserDataResponse(**result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
