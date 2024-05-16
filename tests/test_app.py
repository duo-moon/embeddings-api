from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from app import app
from text2vec_api.vectorizer import ONNXVectorizer
from text2vec_api.vectotization_service import VectorizationService


class TestApp:

    @patch.object(ONNXVectorizer, 'get_config')
    def test_get_config(self, vectorizer):
        vectorizer.get_config.return_value = {'model_type': 'test', 'vocab_size': 1}
        with app.container.vectorizer.override(vectorizer):
            with TestClient(app) as client:
                response = client.get('/vectorizer')

        assert response.status_code == 200
        assert response.json() == {'config': {'model_type': 'test', 'vocab_size': 1}}

    @pytest.mark.asyncio
    @patch.object(VectorizationService, 'vectorize', new_callable=AsyncMock)
    async def test_vectorize_text(self, vectorization_service):
        vectorization_service.vectorize.return_value = [0.0]

        with app.container.vectorization_service.override(vectorization_service):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post('/text2vec', json={'text': 'test'})

        assert response.status_code == 200
        assert response.json() == {'text': 'test', 'vec': [0.0], 'length': 1}
