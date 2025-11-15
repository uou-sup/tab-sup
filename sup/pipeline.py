"""
Utility helpers that connect dataset preprocessing with the discrete graph
construction used by the diffusion model.
"""

from typing import Dict, Tuple

import torch

from .data import Dataset, Transformations, build_dataset, infer_block_group_sizes
from .graph_lib import Graph, get_graph

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    wandb = None  # type: ignore


def _log_dataset_summary(
        dataset: Dataset,
        graph: Graph,
        config,
        transformations: Transformations,
) -> None:
    if wandb is None or wandb.run is None:
        return

    summary: Dict[str, object] = {
        "data/task_type": str(dataset.task_type),
        "data/numeric_features": dataset.n_num_features,
        "data/categorical_features": dataset.n_cat_features,
        "graph/type": getattr(getattr(config, "graph", None), "type", "unknown"),
        "graph/dim": getattr(graph, "dim", 0),
        "transform/normalization": transformations.normalization,
        "transform/cat_encoding": transformations.cat_encoding,
    }

    for split, values in dataset.y.items():
        summary[f"data/rows_{split}"] = len(values)
    if dataset.cat_group_sizes:
        summary["graph/categorical_groups"] = len(dataset.cat_group_sizes)

    wandb.run.summary.update(summary)


def prepare_dataset_and_graph(
        config,
        dataset_path,
        transformations: Transformations,
        device: torch.device,
        cache: bool = True,
        log_to_wandb: bool = False,
) -> Tuple[Dataset, Graph]:
    """
    Loads and transforms the dataset, then instantiates the diffusion graph.

    For BlockUniform graphs, the categorical group sizes are inferred directly
    from the processed dataset so that the transition matrix aligns with the
    tabular feature layout.
    """
    dataset = build_dataset(dataset_path, transformations, cache)

    group_sizes = None
    if getattr(getattr(config, 'graph', None), 'type', None) == 'block_uniform':
        group_sizes = infer_block_group_sizes(dataset)

    graph = get_graph(config, device, group_sizes=group_sizes, dataset=dataset)
    if log_to_wandb:
        _log_dataset_summary(dataset, graph, config, transformations)
    return dataset, graph
