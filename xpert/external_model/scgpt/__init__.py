from .gene_tokenizer import (
    GeneVocab,
    get_default_gene_vocab,
    tokenize_batch,
    pad_batch,
    tokenize_and_pad_batch,
    random_mask_value,
)

# 同时保留已有接口
def load_model(*args, **kwargs):
    raise NotImplementedError

def predict(*args, **kwargs):
    raise NotImplementedError

__all__ = [
    "GeneVocab",
    "get_default_gene_vocab",
    "tokenize_batch",
    "pad_batch",
    "tokenize_and_pad_batch",
    "random_mask_value",
    "load_model",
    "predict",
]