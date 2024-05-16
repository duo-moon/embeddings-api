import asyncio
from concurrent.futures import ThreadPoolExecutor

import torch

from .vectorizer import Vectorizer


class VectorizationService:

    def __init__(self, vectorizer: Vectorizer, executor: ThreadPoolExecutor):
        self.vectorizer = vectorizer
        self.executor = executor

    async def vectorize(self, text: str) -> torch.Tensor:
        return await asyncio.wrap_future(self.executor.submit(self.vectorizer.vectorize, text))
