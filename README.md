# Embeddings API

Embeddings API is a REST API service designed to creates an embedding vector representing the input text.
The API is built to experiment with pre-trained text embedding models. The service supports the ORT model format and a variety of pooling strategies (mean, max).

## Getting Started
### Prerequisites
* **Python 3.11**
* **Poetry 1.8.3**
* **Docker**

### Installation
##### Running the Service Locally
To run the API locally, create and fill an `.env` file (see `.env.example`).
```bash
poetry install
```
```bash
uvicorn --factory embeddings_api.app:build_app --host 0.0.0.0 --port 8080
```

#### Docker Support
You can also run the API using Docker:
```bash
docker build --tag embeddings-api .
```
```bash
docker run -d \
  -it \
  -p 8080:8080 \
  --name embeddings-api \
  --mount type=bind,source="$(pwd)"/<SRC_MOUNT_PATH>,target=/opt/app/<DST_MOUNT_PATH>/ \
  embeddings-api:latest
```

## Usage
#### 1. Health Check
To health check make a GET request to the `/v1/health` endpoint:
```bash
curl -X 'GET' \
  'http://0.0.0.0:8080/v1/health' \
  -H 'accept: application/json'
```
Example response:
```json
{
  "status": "OK"
}
```
#### 2. Generating Text Embeddings
To generate embeddings for a text input, make a POST request to the `/v1/embeddings` endpoint:
```bash
curl -X 'POST' \
  'http://0.0.0.0:8080/v1/embeddings' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "inputs": [
    "hello",
    "world"
  ],
  "pooling": "mean"
}'
```
Example response:
```json
{
  "embeddings": [
    [
      -0.37831297516822815,
      0.3312261402606964,
      0.31438714265823364
    ],
    [
      -0.20730429887771606,
      0.2169610857963562,
      -0.43447697162628174
    ]
  ]
}
```
