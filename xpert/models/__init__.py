"""
Perturbation modeling algorithms and methods.
对外统一导出，便于包外直接引用。
"""

from .base import PerturbationModel

# 传统/基础模型与组件（保留主入口，避免过度污染命名空间）
from .model import TransformerModel  # noqa: F401

# 新增的生成模型接口（用户关注的对外可用入口）
from .generation_model import (
    TransformerGenerator,
    GeneEncoder,
    PositionalEncoding,
    Similarity,
    ClsDecoder,
)  # noqa: F401

from ..loss import masked_mse_loss, masked_relative_error  # noqa: F401

__all__ = [
    # base
    "PerturbationModel",
    # legacy/main transformer model
    "TransformerModel",
    # generation model public API
    "TransformerGenerator",
    "GeneEncoder",
    "PositionalEncoding",
    "Similarity",
    "ClsDecoder",
    # loss
    "masked_mse_loss",
    "masked_relative_error",
]
