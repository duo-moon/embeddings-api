from fastapi import FastAPI

from .config import get_settings
from .container import ApplicationContainer
from .router import router


def build_app() -> FastAPI:
    container = ApplicationContainer()
    settings = get_settings()
    container.config.from_dict(settings)

    app = FastAPI()
    app.container = container
    app.include_router(router)
    return app
