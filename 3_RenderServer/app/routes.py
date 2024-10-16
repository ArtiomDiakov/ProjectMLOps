from fastapi import APIRouter, HTTPException
from app.models import (
    UserRequest,
    RecommendationResponse,
    DeveloperResponse,
    UserDataResponse,
)
from app.utils import (
    recomendacion_usuario,
    developer,
    userdata,
)

router = APIRouter()

@router.post("/recommend", response_model=RecommendationResponse)
def recommend_games(user_request: UserRequest):
    user_id = user_request.user_id
    recommendations = recomendacion_usuario(user_id)
    if recommendations is None or not isinstance(recommendations, list):
        raise HTTPException(status_code=404, detail=f"No recommendations could be generated for user {user_id}.")
    if not recommendations:
        return RecommendationResponse(user_id=user_id, recommended_games=[])
    return RecommendationResponse(user_id=user_id, recommended_games=recommendations)

@router.get("/developer", response_model=DeveloperResponse)
def get_developer_stats(desarrollador: str):
    stats = developer(desarrollador)
    if stats is None or len(stats) == 0:
        raise HTTPException(status_code=404, detail=f"No data found for developer: {desarrollador}")
    return DeveloperResponse(developer=desarrollador, stats=stats)


@router.get("/userdata", response_model=UserDataResponse)
def get_user_data(user_id: str):
    result = userdata(user_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found.")
    return UserDataResponse(**result)
