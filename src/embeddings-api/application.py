from fastapi import FastAPI

from .config import get_settings
from .container import ApplicationContainer
from .router import router


def create_app() -> FastAPI:
    container = ApplicationContainer()
    container.config.from_dict(get_settings().model_dump())
    container.init_resources()

    app = FastAPI()
    app.container = container
    app.include_router(router)
    return app
