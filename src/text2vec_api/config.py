import pathlib
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env')

    vec_model_path: pathlib.Path
    max_workers: int


@lru_cache()
def get_settings() -> Settings:
    return Settings()
