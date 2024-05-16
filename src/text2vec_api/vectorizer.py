import abc
import pathlib

import torch
import torch.nn.functional as F
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer


class Vectorizer(abc.ABC):
    @abc.abstractmethod
    def vectorize(self, text: str) -> torch.Tensor:
        pass

    def get_config(self) -> dict[str]:
        ...


class ONNXVectorizer(Vectorizer):
    model: ORTModelForFeatureExtraction
    tokenizer: AutoTokenizer

    def __init__(self, model_path: pathlib.Path):
        self.model = ORTModelForFeatureExtraction.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)

    def vectorize(self, text: str) -> torch.Tensor:
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            output = self.model(**encoded_input)

        embeddings = self.mean_pooling(model_output=output, attention_mask=encoded_input['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings[0]

    def get_config(self) -> dict:
        return self.model.config.to_dict()

    @staticmethod
    def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
