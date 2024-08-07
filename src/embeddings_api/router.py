from typing import Any, Optional
import typing
from dependency_injector.wiring import inject, Provide
from fastapi import APIRouter, Depends, status
from pydantic import BaseModel

from .app import ApplicationContainer
from .vectorizer import Vectorizer


class HealthCheck(BaseModel):
    status: str


class VectorizationRequest(BaseModel):
    inputs: list[str]
    pooling: str


class VectorizationResponse(BaseModel):
    embeddings: list[list[float]]


router = APIRouter()


@router.get('/v1/health', status_code=status.HTTP_200_OK, response_model=HealthCheck)
async def health():
    return {'status': 'OK'}


@router.post('/v1/embeddings', response_model=VectorizationResponse)
@inject
async def vectorize(
        request: VectorizationRequest,
        vectorizer: Vectorizer = Depends(Provide[ApplicationContainer.vectorizer])
) -> typing.Any:
    results = vectorizer.vectorize(inputs=request.inputs, pooling_mode=request.pooling)
    return {
        'embeddings': results,
    }
