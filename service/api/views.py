from typing import List

from fastapi import APIRouter, FastAPI, Request
from pydantic import BaseModel

from service.api.constants import AVAILABLE_MODEL_NAMES
from service.api.exceptions import ModelNotFoundError, UserNotFoundError
from service.log import app_logger

from service.api.recommender import Recommender

class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


router = APIRouter()
recommender = Recommender(dataset_path = "artifacts/interactions.csv",
                          warm_model_path = "artifacts/first_experiment_popular.pkl",
                          hot_model_path = "artifacts/task3_cropped20_experiment_tfidf_userknn.pkl")

@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    # Проверка на корректность имени модели
    if model_name not in AVAILABLE_MODEL_NAMES:
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")

    # Проверка на корректность индекса пользователя
    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    k_recs = request.app.state.k_recs
    reco = recommender.recommend(user_id, k_recs)
    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
