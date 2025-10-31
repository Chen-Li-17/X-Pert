"""
GEARS external model utilities.

Direct imports:
    from xpert.external_model.gears import evaluate, compute_metrics, batch_predict
"""

from .inference import (
    evaluate,
    compute_metrics,
    non_zero_analysis,
    non_dropout_analysis,
    deeper_analysis,
    GI_subgroup,
    node_specific_batch_out,
    batch_predict,
    get_high_umi_idx,
    get_mean_ctrl,
    get_single_name,
    get_test_set_results_seen2,
    get_all_vectors,
)

__all__ = [
    "evaluate",
    "compute_metrics",
    "non_zero_analysis",
    "non_dropout_analysis",
    "deeper_analysis",
    "GI_subgroup",
    "node_specific_batch_out",
    "batch_predict",
    "get_high_umi_idx",
    "get_mean_ctrl",
    "get_single_name",
    "get_test_set_results_seen2",
    "get_all_vectors",
]


