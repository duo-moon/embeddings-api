import torch
from transformers.modeling_outputs import BaseModelOutput


def mean_pooling(model_output: BaseModelOutput, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def max_pooling(model_output: BaseModelOutput, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    token_embeddings[input_mask_expanded == 0] = -1e9
    return torch.max(token_embeddings, 1)[0]
