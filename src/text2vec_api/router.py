from typing import Any

from dependency_injector.wiring import inject, Provide
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from .application import ApplicationContainer
from .vectorizer import Vectorizer
from .vectotization_service import VectorizationService


class InputText(BaseModel):
    text: str


class Text2Vec(BaseModel):
    text: str
    vec: list[float]
    length: int


class VectorizerConfig(BaseModel):
    config: dict


router = APIRouter()


@router.get("/vectorizer", response_model=VectorizerConfig)
@inject
async def get_config(vectorizer: Vectorizer = Depends(Provide[ApplicationContainer.vectorizer])) -> Any:
    return {'config': vectorizer.get_config()}


@router.post("/text2vec", response_model=Text2Vec)
@inject
async def vectorize_text(
        input_text: InputText,
        vectorization_srvice: VectorizationService = Depends(Provide[ApplicationContainer.vectorization_service])
) -> Any:
    result = await vectorization_srvice.vectorize(text=input_text.text)
    return {
        'text': input_text.text,
        'vec': result,
        'length': len(result)
    }
