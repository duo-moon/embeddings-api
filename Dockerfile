FROM python:3.11-buster as builder

RUN pip install poetry==1.8.2

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/opt/.cache \
    POETRY_VENV=/opt/app/.venv

WORKDIR /opt/app/

COPY pyproject.toml poetry.lock ./
RUN touch README.md

RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR

FROM python:3.11-slim-buster as app

ENV PATH="${POETRY_VENV}/bin:$PATH"

COPY --from=builder ${POETRY_VENV} ${POETRY_VENV}

WORKDIR /opt/app/
COPY /src/. app.py .env ./
