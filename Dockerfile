FROM python:3.11-buster

RUN pip install poetry==1.8.2

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/opt/.cache \
    POETRY_VENV=/opt/app/.venv

WORKDIR /opt/app/

COPY ./pyproject.toml ./poetry.lock ./

RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR

ENV PATH="${POETRY_VENV}/bin:$PATH"

COPY ./src/. ./
COPY /.env ./

CMD ["uvicorn", "--factory", "embeddings_api.app:build_app", "--host", "0.0.0.0", "--port", "8080"]
