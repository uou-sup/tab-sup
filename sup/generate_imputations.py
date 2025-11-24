import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from . import utils
from .ema import ExponentialMovingAverage
from .model.impute import impute_block_absorbing, impute_block_uniform
from .noise_lib import get_noise
from .pipeline import prepare_dataset_and_graph
from .sampling import decoded_to_dataframe
from .train import Config, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Impute missing entries using a trained Tab-SEDD checkpoint."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the training config TOML file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=False,
        help="Checkpoint file to load (defaults to generation.checkpoint).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device string (default: auto cuda/cpu detection).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("train", "val", "test"),
        default=None,
        help="Optional dataset split to pull inputs from (before masking).",
    )
    parser.add_argument(
        "--row-start",
        type=int,
        default=0,
        help="Starting row index when slicing the chosen split or array.",
    )
    parser.add_argument(
        "--row-count",
        type=int,
        default=None,
        help="Number of rows to process (default: until the end).",
    )
    parser.add_argument(
        "--tokens",
        type=Path,
        default=None,
        help="Optional path to a .npy file containing local categorical tokens.",
    )
    parser.add_argument(
        "--token-mask",
        type=Path,
        default=None,
        help="Optional .npy file with a boolean mask (True -> needs imputation) for categorical features.",
    )
    parser.add_argument(
        "--token-mask-value",
        type=int,
        default=None,
        help="If provided, infer the categorical mask wherever tokens equal this value.",
    )
    parser.add_argument(
        "--numeric",
        type=Path,
        default=None,
        help="Optional path to a .npy file containing numerical features.",
    )
    parser.add_argument(
        "--numeric-mask",
        type=Path,
        default=None,
        help="Optional .npy file with a boolean mask (True -> needs imputation) for numerical features.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of predictor steps (default: config.sampling.steps).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-5,
        help="Terminal time for the sampler (default: 1e-5).",
    )
    parser.add_argument(
        "--numeric-clip",
        type=float,
        default=None,
        help="Optional clamp value applied to numeric states after each update.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("samples/imputations"),
        help="Directory where imputed arrays will be saved.",
    )
    parser.add_argument(
        "--decoded-output",
        type=Path,
        default=None,
        help="Optional CSV path for decoded outputs (categorical + numeric).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = utils.from_dict(Config, utils.load_config(args.config))
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    dataset, graph = prepare_dataset_and_graph(
        config,
        config.dataset.path,
        config.transformations,
        device,
        cache=config.dataset.cache,
        log_to_wandb=False,
    )

    input_dim = graph.dim + dataset.n_num_features
    model = build_model(config.model, dataset, input_dim).to(device)
    num_numerical = dataset.n_num_features if dataset.n_num_features > 0 else None
    num_categorical = dataset.n_cat_features if dataset.n_cat_features > 0 else None
    noise = get_noise(
        config.noise,
        config.numeric_noise,
        num_numerical=num_numerical,
        num_categorical=num_categorical,
    )

    checkpoint_path = args.checkpoint or (
        Path(config.generation.checkpoint)
        if config.generation and config.generation.checkpoint
        else None
    )
    if checkpoint_path is None:
        raise ValueError("No checkpoint specified. Provide --checkpoint or set generation.checkpoint in the config.")
    checkpoint_path = Path(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    ema = ExponentialMovingAverage(model.parameters(), config.train.ema_decay)
    ema.load_state_dict(checkpoint["ema"])
    ema.copy_to(model.parameters())
    model.eval()

    tokens_np = _resolve_array(args.tokens, dataset.X_cat, args.split, args.row_start, args.row_count)
    numeric_np = _resolve_array(args.numeric, dataset.X_num, args.split, args.row_start, args.row_count)

    if tokens_np is None and numeric_np is None:
        raise ValueError("At least one of categorical tokens or numeric values must be provided.")

    tokens = torch.from_numpy(tokens_np).long() if tokens_np is not None else None
    numeric = torch.from_numpy(numeric_np).float() if numeric_np is not None else None

    token_mask = _resolve_token_mask(tokens, args.token_mask, args.token_mask_value, graph, args.row_start, args.row_count)
    numeric_mask = _resolve_numeric_mask(numeric, args.numeric_mask, args.row_start, args.row_count)

    steps = args.steps or config.sampling.steps
    numeric_clip = args.numeric_clip
    if numeric_clip is None and config.generation and config.generation.numeric_clip:
        numeric_clip = config.generation.numeric_clip

    if getattr(graph, "absorb", False):
        imputed_tokens, imputed_numeric = impute_block_absorbing(
            model=model,
            graph=graph,
            noise=noise,
            tokens=tokens,
            token_mask=token_mask,
            numeric=numeric,
            numeric_mask=numeric_mask,
            steps=steps,
            eps=args.eps,
            numeric_clip=numeric_clip,
            device=device,
        )
    else:
        imputed_tokens, imputed_numeric = impute_block_uniform(
            model=model,
            graph=graph,
            noise=noise,
            tokens=tokens,
            token_mask=token_mask,
            numeric=numeric,
            numeric_mask=numeric_mask,
            steps=steps,
            eps=args.eps,
            numeric_clip=numeric_clip,
            device=device,
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if imputed_tokens is not None:
        tokens_np = imputed_tokens.detach().cpu().numpy()
        np.save(output_dir / "imputed_tokens.npy", tokens_np)
    if imputed_numeric is not None:
        numeric_np = imputed_numeric.detach().cpu().numpy()
        np.save(output_dir / "imputed_numeric.npy", numeric_np)

    if args.decoded_output:
        decoded = _decode_categorical(imputed_tokens, dataset, graph)
        numeric_decoded = _decode_numeric(imputed_numeric, dataset)
        info_path = Path(config.dataset.path) / "info.json"
        info = utils.load_json(info_path) if info_path.exists() else {}
        df = decoded_to_dataframe(
            decoded,
            columns=info.get("categorical_columns"),
            numeric=numeric_decoded,
            numeric_columns=info.get("numeric_columns"),
        )
        df.to_csv(args.decoded_output, index=False)
        print(f"Decoded imputation saved to {args.decoded_output}")

    print(f"Imputation artifacts saved to {output_dir.resolve()}")


def _resolve_array(
    override_path: Optional[Path],
    dataset_split: Optional[dict],
    split: Optional[str],
    start: int,
    count: Optional[int],
) -> Optional[np.ndarray]:
    if override_path is not None:
        array = np.load(override_path)
    elif split is not None and dataset_split is not None:
        if split not in dataset_split:
            raise ValueError(f"Split '{split}' not available in dataset.")
        array = dataset_split[split]
    else:
        return None
    if array.ndim == 1:
        array = np.expand_dims(array, axis=0)
    end = start + count if count is not None else None
    return array[start:end]


def _resolve_token_mask(
    tokens: Optional[torch.Tensor],
    mask_path: Optional[Path],
    mask_value: Optional[int],
    graph,
    start: int,
    count: Optional[int],
) -> Optional[torch.Tensor]:
    if tokens is None:
        return None
    if mask_path is not None:
        mask_np = np.load(mask_path)
        if mask_np.ndim == 1:
            mask_np = np.expand_dims(mask_np, axis=0)
        end = start + count if count is not None else None
        mask_np = mask_np[start:end]
        mask = torch.from_numpy(mask_np.astype(bool))
    elif mask_value is not None:
        mask = (tokens == mask_value)
    elif getattr(graph, "absorb", False):
        mask = _infer_absorbing_mask(tokens, graph)
    else:
        raise ValueError("token_mask is required for categorical imputation without absorbing states.")
    if mask.shape != tokens.shape:
        raise ValueError(f"token_mask shape {tuple(mask.shape)} does not match tokens shape {tuple(tokens.shape)}.")
    return mask.bool()


def _resolve_numeric_mask(
    numeric: Optional[torch.Tensor],
    mask_path: Optional[Path],
    start: int,
    count: Optional[int],
) -> Optional[torch.Tensor]:
    if numeric is None:
        return None
    if mask_path is not None:
        mask_np = np.load(mask_path)
        if mask_np.ndim == 1:
            mask_np = np.expand_dims(mask_np, axis=0)
        end = start + count if count is not None else None
        mask_np = mask_np[start:end]
        mask = torch.from_numpy(mask_np.astype(bool))
    else:
        mask = torch.isnan(numeric)
    if mask.shape != numeric.shape:
        raise ValueError(f"numeric_mask shape {tuple(mask.shape)} does not match numeric shape {tuple(numeric.shape)}.")
    return mask.bool()


def _infer_absorbing_mask(tokens: torch.Tensor, graph) -> torch.Tensor:
    base_sizes = getattr(graph, "base_group_sizes", None)
    if base_sizes is None:
        raise ValueError("Cannot infer absorbing mask without base_group_sizes metadata.")
    mask_cols = []
    for idx, base_size in enumerate(base_sizes):
        mask_cols.append(tokens[:, idx] >= base_size)
    return torch.stack(mask_cols, dim=1)


def _decode_categorical(
    tokens: Optional[torch.Tensor],
    dataset,
    graph,
) -> Optional[np.ndarray]:
    if tokens is None:
        return None
    local = tokens.detach().cpu().numpy()
    group_sizes = dataset.cat_group_sizes or getattr(graph, "base_group_sizes", None)
    if group_sizes is not None:
        for idx, size in enumerate(group_sizes):
            local[:, idx] = np.clip(local[:, idx], 0, size - 1)
    if dataset.cat_transform is not None:
        return dataset.cat_transform.inverse_transform(local)
    return local


def _decode_numeric(
    numeric: Optional[torch.Tensor],
    dataset,
) -> Optional[np.ndarray]:
    if numeric is None:
        return None
    numeric_np = numeric.detach().cpu().numpy()
    if dataset.num_transform is not None:
        numeric_np = dataset.num_transform.inverse_transform(numeric_np)
    if getattr(dataset, "dequantizer", None) is not None:
        numeric_np = dataset.dequantizer.inverse_transform(numeric_np)
    return numeric_np


if __name__ == "__main__":
    main()
