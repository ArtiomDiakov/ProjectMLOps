from pydantic import BaseModel
from typing import List

class UserRequest(BaseModel):
    user_id: str

class RecommendationResponse(BaseModel):
    user_id: str
    recommended_games: List[str]

class DeveloperStats(BaseModel):
    year: int
    cantidad_items: int
    contenido_free: float

class DeveloperResponse(BaseModel):
    developer: str
    stats: List[DeveloperStats]

class UserDataResponse(BaseModel):
    usuario: str
    dinero_gastado: str
    porcentaje_recomendacion: str
    cantidad_items: int