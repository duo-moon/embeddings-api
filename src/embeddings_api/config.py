import pathlib

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    device: str
    vectorization_model_path: pathlib.Path
    tokenizer_path: pathlib.Path

    model_config = SettingsConfigDict(env_file='.env')


def get_settings() -> dict:
    return Settings().model_dump()
