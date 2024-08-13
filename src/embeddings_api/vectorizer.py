import abc
import pathlib

import torch
from optimum.modeling_base import PreTrainedModel
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer, PreTrainedTokenizer

from .helpers import mean_pooling, max_pooling


class Vectorizer(abc.ABC):
    @abc.abstractmethod
    def vectorize(self, inputs: list[str], pooling_mode: str) -> list[list[float]]:
        pass


class ORTPreTrainedModelVectorizer(Vectorizer):
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    pooling_strategies = {
        'mean': mean_pooling,
        'max': max_pooling,
    }

    def __init__(self, device: str, model_path: pathlib.Path, tokenizer_path: pathlib.Path):
        self.model = ORTModelForFeatureExtraction.from_pretrained(model_path, local_files_only=True, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, device=device)

    def vectorize(self, inputs: list[str], pooling_mode: str) -> list[list[float]]:
        if pooling_mode not in self.pooling_strategies:
            raise ValueError('pooling_mode must be one of {}'.format(', '.join(self.pooling_strategies)))

        encoded_input = self.tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        embeddings = self.pooling_strategies[pooling_mode](model_output, encoded_input['attention_mask'])

        return embeddings.tolist()
