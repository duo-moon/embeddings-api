from concurrent.futures import ThreadPoolExecutor

from dependency_injector import containers, providers

from .vectorizer import ONNXVectorizer
from .vectotization_service import VectorizationService


def init_thread_pool(max_workers: int):
    thread_pool = ThreadPoolExecutor(max_workers=max_workers)
    yield thread_pool
    thread_pool.shutdown(wait=True)


class ApplicationContainer(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(modules=[".router"])

    config = providers.Configuration()

    thread_pool = providers.Resource(
        init_thread_pool,
        max_workers=config.max_workers,
    )

    vectorizer = providers.Factory(
        ONNXVectorizer,
        model_path=config.vec_model_path,
    )

    vectorization_service = providers.Factory(
        VectorizationService,
        executor=thread_pool,
        vectorizer=vectorizer,
    )
