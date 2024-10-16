from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

from typing import List
# Importar otras librerías necesarias

###############################################################################################################
### User-Item Recommendations

# Cargar DataFrames
df_user_game = pd.read_parquet('df_user_game.parquet')
similarity_df = pd.read_parquet('similarity_df.parquet')
df_filtered_games = pd.read_parquet('df_filtered_games.parquet')
df_exploded = pd.read_parquet('df_exploded.parquet')

# Cargar modelos y estructuras necesarias
cosine_sim_games = joblib.load('cosine_sim_games.pkl')
indices_games = joblib.load('indices_games.pkl')

app = FastAPI()

class UserRequest(BaseModel):
    user_id: str

class RecommendationResponse(BaseModel):
    user_id: str
    recommended_games: list

def recomendacion_contenido(game_name, cosine_sim=cosine_sim_games):
    # Verificar si el juego existe en el índice
    if game_name not in indices_games:
        return []

    # Obtener el índice del juego
    idx = indices_games[game_name]

    # Obtener las puntuaciones de similitud para todos los juegos con respecto al juego dado
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Ordenar los juegos por puntuación de similitud
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Obtener los índices de los juegos más similares
    sim_scores = sim_scores[1:6]  # Ignorar el primero ya que es el mismo juego

    # Obtener los índices de los juegos recomendados
    game_indices = [i[0] for i in sim_scores]

    # Devolver los nombres de los juegos recomendados
    return df_filtered_games['app_name'].iloc[game_indices].tolist()


def recomendacion_usuario(user_id, df_user_game, similarity_df, df_exploded, df_filtered_games, num_recommendations=5):
    if user_id in similarity_df.index:
        # Collaborative filtering recommendations
        similar_users = similarity_df[user_id].sort_values(ascending=False)
        similar_users = similar_users.drop(labels=[user_id])
        top_similar_users = similar_users.head(5).index
        recommended_games = df_user_game[df_user_game['user_id'].isin(top_similar_users)]['game_id'].tolist()
        user_games = df_user_game[df_user_game['user_id'] == user_id]['game_id'].tolist()
        recommended_games = [game for game in recommended_games if game not in user_games]
        recommended_games_series = pd.Series(recommended_games).value_counts()
        recommended_games = recommended_games_series.head(num_recommendations).index.tolist()
        # Get game names
        recommended_game_names = df_filtered_games[df_filtered_games['id'].astype(str).isin(recommended_games)]['app_name'].unique().tolist()
        return recommended_game_names
    else:
        # Content-based recommendations
        user_games = df_user_game[df_user_game['user_id'] == user_id]['game_id'].tolist()
        if not user_games:
            # User has not played any games or does not exist in df_user_game
            return ["user_not_found"]
        user_game_names = df_filtered_games[df_filtered_games['id'].astype(str).isin(user_games)]['app_name'].tolist()
        if not user_game_names:
            # No matching games found in df_filtered_games
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
    recommendations = recomendacion_usuario(user_id, df_user_game, similarity_df, df_exploded, df_filtered_games, num_recommendations=5)
    if recommendations is None or not isinstance(recommendations, list):
        raise HTTPException(status_code=404, detail=f"No recommendations could be generated for user {user_id}.")
    if not recommendations:
        # If the list is empty, return a message or an empty list
        return RecommendationResponse(user_id=user_id, recommended_games=[])
    return RecommendationResponse(user_id=user_id, recommended_games=recommendations)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

###################################################################################################################################
# def developer( desarrollador : str ): Cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora. 

# Asegurarnos de que df_filtered_games está cargado
    df_filtered_games = pd.read_parquet('fixed_steam_games.parquet')

# Crear la columna 'free' si no existe
if 'free' not in df_filtered_games.columns:
    df_filtered_games['free'] = df_filtered_games['price'].apply(lambda x: True if x == 0 else False)

# Convertir 'release_date' a tipo datetime si no lo está
df_filtered_games['release_date'] = pd.to_datetime(df_filtered_games['release_date'], errors='coerce')

# Crear una columna 'year' con el año de lanzamiento
df_filtered_games['year'] = df_filtered_games['release_date'].dt.year


def developer(desarrollador: str):
    # Filtrar los juegos del desarrollador dado
    df_dev = df_filtered_games[df_filtered_games['developer'].str.contains(desarrollador, case=False, na=False)].copy()
    
    if df_dev.empty:
        return None  # O puedes devolver un mensaje indicando que no se encontraron resultados
    
    # Agrupar por año y calcular la cantidad de items y el porcentaje de contenido gratuito
    df_summary = df_dev.groupby('year').agg(
        cantidad_items=('id', 'count'),
        contenido_free=('free', lambda x: round((x.sum() / x.count()) * 100, 2))
    ).reset_index()
    
    # Ordenar por año descendente
    df_summary = df_summary.sort_values(by='year', ascending=False)
    
    # Convertir a lista de diccionarios
    result = df_summary.to_dict(orient='records')
    
    return result

class DeveloperStats(BaseModel):
    year: int
    cantidad_items: int
    contenido_free: float  # Usaremos float para representar el porcentaje

class DeveloperResponse(BaseModel):
    developer: str
    stats: List[DeveloperStats]

@app.get("/developer", response_model=DeveloperResponse)
def get_developer_stats(desarrollador: str):
    stats = developer(desarrollador)
    if stats is None or len(stats) == 0:
        raise HTTPException(status_code=404, detail=f"No se encontraron datos para el desarrollador: {desarrollador}")
    return DeveloperResponse(developer=desarrollador, stats=stats)

# Check if user_id exists in df_user_game
if "-PRoSlayeR-" in df_user_game['user_id'].values:
    print("User found in df_user_game.")
else:
    print("User not found in df_user_game.")

#########################################################################################################################################################
#def userdata( User_id : str ): Debe devolver cantidad de dinero gastado por el usuario, el porcentaje de recomendación en base a reviews.recommend y cantidad de items.

# Cargar fixed_users_items.parquet
df_users_items = pd.read_parquet('fixed_users_items.parquet')

# Cargar fixed_steam_games.parquet
df_steam_games = pd.read_parquet('fixed_steam_games.parquet')

# Cargar user_reviews_with_sentiment.parquet
df_user_reviews = pd.read_parquet('user_reviews_with_sentiment.parquet')



class UserDataResponse(BaseModel):
    usuario: str
    dinero_gastado: str
    porcentaje_recomendacion: str
    cantidad_items: int

def userdata(user_id: str):
    # Verify if the user exists in df_users_items
    user_data = df_users_items[df_users_items['user_id'] == user_id]
    if user_data.empty:
        return None  # Or raise an exception

    # Get the number of items
    cantidad_items = user_data.iloc[0]['items_count']

    # Extract the user's items
    user_items = user_data.iloc[0]['items']

    # Initialize dinero_gastado_str
    dinero_gastado_str = "0.00 USD"

    # Check if user_items is None or empty
    if user_items is None or len(user_items) == 0:
        print(f"The 'items' field is empty for user '{user_id}'.")
        # dinero_gastado_str remains "0.00 USD"
    else:
        print(f"The 'items' field for user '{user_id}' contains data.")

        # Use pd.json_normalize to create the DataFrame
        df_user_items = pd.json_normalize(user_items)

        # Debugging: Check columns and data
        print("Columns in df_user_items:", df_user_items.columns.tolist())
        print("Head of df_user_items:")
        print(df_user_items.head())

        # Detect the correct column for item IDs
        if 'item_id' in df_user_items.columns:
            item_id_column = 'item_id'
        elif 'appid' in df_user_items.columns:
            item_id_column = 'appid'
        else:
            print("No 'item_id' or 'appid' column found in df_user_items.")
            print(f"Available columns: {df_user_items.columns.tolist()}")
            # Handle appropriately
            dinero_gastado_str = "0.00 USD"
            item_id_column = None

        if item_id_column:
            # Ensure that item_id_column is of type integer
            df_user_items[item_id_column] = df_user_items[item_id_column].astype(int)

            # Merge with df_steam_games to get prices
            df_user_items = df_user_items.merge(
                df_steam_games[['id', 'price']], left_on=item_id_column, right_on='id', how='left'
            )

            # Replace null prices with 0
            df_user_items['price'] = df_user_items['price'].fillna(0)

            # Calculate total spent (sum of prices)
            dinero_gastado = df_user_items['price'].sum()

            # Format the total spent
            dinero_gastado_str = f"{dinero_gastado:.2f} USD"

    # Get the user's reviews
    user_reviews = df_user_reviews[df_user_reviews['user_id'] == user_id]

    if not user_reviews.empty:
        # Calculate the percentage of positive recommendations
        porcentaje_recomendacion = (user_reviews['recommend'].sum() / user_reviews.shape[0]) * 100
        porcentaje_recomendacion_str = f"{porcentaje_recomendacion:.2f}%"
    else:
        porcentaje_recomendacion_str = "No hay reviews disponibles"

    # Prepare the response
    response = {
        "usuario": user_id,
        "dinero_gastado": dinero_gastado_str,
        "porcentaje_recomendacion": porcentaje_recomendacion_str,
        "cantidad_items": cantidad_items
    }

    return response

@app.get("/userdata", response_model=UserDataResponse)
def get_user_data(user_id: str):
    result = userdata(user_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Usuario {user_id} no encontrado.")
    return UserDataResponse(**result)

