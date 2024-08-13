from dependency_injector import containers, providers

from .vectorizer import ORTPreTrainedModelVectorizer


class ApplicationContainer(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(modules=[".router"])

    config = providers.Configuration()

    vectorizer = providers.Factory(
        ORTPreTrainedModelVectorizer,
        device=config.device,
        model_path=config.vectorization_model_path,
        tokenizer_path=config.tokenizer_path,
    )
