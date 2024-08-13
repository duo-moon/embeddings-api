from unittest.mock import patch

from fastapi.testclient import TestClient

from embeddings_api.app import build_app
from embeddings_api.vectorizer import Vectorizer

app = build_app()

class TestApp:

    @patch.object(Vectorizer, 'vectorize')
    def test_vectorize_text(self, vectorizer_mock):
        vectorizer_mock.vectorize.return_value = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        with app.container.vectorizer.override(vectorizer_mock):
            with TestClient(app) as client:
                response = client.post('/v1/embeddings', json={'inputs': ['hello', 'world'], 'pooling': 'mean'})

        assert response.status_code == 200
        assert response.json() == {'embeddings': [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]}
