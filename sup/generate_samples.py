import argparse
from pathlib import Path

import numpy as np
import torch

from . import utils
from .ema import ExponentialMovingAverage
from .noise_lib import get_noise
from .pipeline import prepare_dataset_and_graph
from .sampling import (
    decode_block_uniform_tokens,
    decoded_to_dataframe,
    sample_block_uniform,
)
from .train import Config, build_model, GenerationConfig
from .utils import TaskType

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    wandb = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic samples from a trained Tab-SEDD checkpoint."
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
        help="Checkpoint file to load (e.g. ).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the generated CSV (default: samples/generated.csv).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of synthetic rows to generate (default: config or 512).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device string (default: auto cuda/cpu detection).",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Log sampling metadata, previews, and CSV artifacts to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Optional WandB project name (defaults to 'tab-sup').",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional WandB run name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = utils.from_dict(Config, utils.load_config(args.config))
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    generation_cfg: GenerationConfig = config.generation or GenerationConfig()
    checkpoint_path = args.checkpoint or (
        Path(generation_cfg.checkpoint) if generation_cfg.checkpoint else None
    )
    if checkpoint_path is None:
        raise ValueError("No checkpoint specified. Provide --checkpoint or set generation.checkpoint in the config.")
    checkpoint_path = Path(checkpoint_path)

    num_samples = args.num_samples if args.num_samples is not None else generation_cfg.num_samples
    if num_samples is None:
        num_samples = 512
    output_path = args.output or (Path(generation_cfg.output) if generation_cfg.output else Path("samples/generated.csv"))
    output_path = Path(output_path)

    wandb_run = None
    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is requested but not installed. Install it with `pip install wandb`.")
        wandb_run = wandb.init(
            project=args.wandb_project or "tab-sup",
            name=args.wandb_run_name,
            job_type="sampling",
            config={
                "num_samples": num_samples,
                "sampling_steps": config.sampling.steps,
                "sampling_predictor": config.sampling.predictor,
                "checkpoint": str(checkpoint_path),
                "generation_numeric_clip": generation_cfg.numeric_clip,
            },
        )

    dataset, graph = prepare_dataset_and_graph(
        config,
        config.dataset.path,
        config.transformations,
        device,
        cache=config.dataset.cache,
        log_to_wandb=wandb_run is not None,
    )

    input_dim = graph.dim + dataset.n_num_features
    model = build_model(config.model, dataset, input_dim).to(device)
    num_numerical = dataset.n_num_features if dataset.n_num_features > 0 else None
    noise = get_noise(config.noise, config.numeric_noise, num_numerical=num_numerical)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    ema = ExponentialMovingAverage(model.parameters(), config.train.ema_decay)
    ema.load_state_dict(checkpoint["ema"])
    ema.copy_to(model.parameters())
    model.eval()

    samples = sample_block_uniform(
        model=model,
        graph=graph,
        noise=noise,
        num_samples=num_samples,
        steps=config.sampling.steps,
        predictor=config.sampling.predictor,
        denoise=config.sampling.noise_removal,
        eps=1e-5,
        device=device,
        numeric_clip=generation_cfg.numeric_clip,
    )

    if isinstance(samples, tuple):
        tokens, numeric = samples
    else:
        tokens, numeric = samples, None

    categorical = decode_block_uniform_tokens(graph, tokens, dataset.cat_transform)
    info_path = Path(config.dataset.path) / "info.json"
    info = utils.load_json(info_path) if info_path.exists() else {}
    numeric_np = None
    if numeric is not None:
        clip_value = generation_cfg.numeric_clip
        if clip_value is not None and clip_value > 0:
            numeric = numeric.clamp_(-clip_value, clip_value)
        numeric_np = numeric.detach().cpu().numpy()
        if dataset.num_transform is not None:
            numeric_np = dataset.num_transform.inverse_transform(numeric_np)
        if getattr(dataset, "dequantizer", None) is not None:
            numeric_np = dataset.dequantizer.inverse_transform(numeric_np)
        numeric_cols_meta = info.get("numeric_columns") if info else None
        if numeric_cols_meta and numeric_np.shape[1] > len(numeric_cols_meta):
            extra = numeric_np.shape[1] - len(numeric_cols_meta)
            print(
                f"Trimming {extra} one-hot-expanded numeric columns; keeping first {len(numeric_cols_meta)} original numeric features."
            )
            numeric_np = numeric_np[:, :len(numeric_cols_meta)]
        elif numeric_cols_meta and numeric_np.shape[1] < len(numeric_cols_meta):
            print(
                f"Warning: generated numeric features ({numeric_np.shape[1]}) fewer than metadata columns ({len(numeric_cols_meta)})."
            )

    df = decoded_to_dataframe(
        categorical,
        columns=info.get("categorical_columns"),
        numeric=numeric_np,
        numeric_columns=info.get("numeric_columns"),
    )

    target_column = info.get("target")
    if target_column:
        labels: list[np.ndarray] = []
        for split in ("train", "val", "test"):
            if split in dataset.y:
                labels.append(dataset.y[split])
        if labels:
            y_all = np.concatenate(labels)
            if dataset.task_type in {TaskType.BINCLASS, TaskType.MULTICLASS}:
                y_all_int = y_all.astype(int)
                counts = np.bincount(y_all_int)
                probs = counts / counts.sum()
                rng = np.random.default_rng()
                sampled = rng.choice(len(probs), size=num_samples, p=probs)
                if dataset.task_type == TaskType.BINCLASS:
                    df[target_column] = sampled.astype(bool)
                else:
                    df[target_column] = sampled.astype(int)
            else:
                mean = float(y_all.mean())
                std = float(y_all.std())
                rng = np.random.default_rng()
                sampled = rng.normal(loc=mean, scale=std if std > 0 else 1e-6, size=num_samples)
                if dataset.y_info and "mean" in dataset.y_info and "std" in dataset.y_info:
                    sampled = sampled * dataset.y_info["std"] + dataset.y_info["mean"]
                df[target_column] = sampled

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} samples to {output_path}")

    if wandb_run is not None:
        preview = df.head(min(len(df), 50))
        log_payload = {
            "sampling/num_samples": len(df),
            "sampling/checkpoint": str(checkpoint_path),
        }
        if not preview.empty:
            log_payload["sampling/preview"] = wandb.Table(dataframe=preview)
        wandb.log(log_payload)

        artifact_name = f"samples-{output_path.stem}-{wandb_run.id}"
        artifact = wandb.Artifact(
            name=artifact_name,
            type="generated-data",
            description="Synthetic samples generated via tab-sup",
        )
        artifact.add_file(str(output_path))
        wandb.log_artifact(artifact)
        wandb_run.finish()


if __name__ == "__main__":
    main()
