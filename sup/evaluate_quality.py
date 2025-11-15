import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    wandb = None  # type: ignore


def load_info(info_path: Optional[Path]) -> Dict[str, List[str]]:
    if info_path is None or not info_path.exists():
        return {}
    data = json.loads(info_path.read_text())
    return {
        "categorical_columns": data.get("categorical_columns"),
        "numeric_columns": data.get("numeric_columns"),
        "target": data.get("target"),
    }


def detect_column_types(df: pd.DataFrame, info: Dict[str, List[str]]) -> Tuple[List[str], List[str], Optional[str]]:
    cat_cols = info.get("categorical_columns")
    num_cols = info.get("numeric_columns")
    target = info.get("target")

    if cat_cols is None or num_cols is None:
        inferred_cat: List[str] = []
        inferred_num: List[str] = []
        for col in df.columns:
            if col == target:
                continue
            if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                inferred_num.append(col)
            else:
                inferred_cat.append(col)
        if cat_cols is None:
            cat_cols = inferred_cat
        if num_cols is None:
            num_cols = inferred_num

    return list(cat_cols or []), list(num_cols or []), target


def normalize_counts(series: pd.Series) -> Dict[str, float]:
    counts = series.value_counts(dropna=False)
    total = counts.sum()
    if total == 0:
        return {}
    return (counts / total).to_dict()


def categorical_shape(real: pd.Series, synth: pd.Series) -> float:
    real_probs = normalize_counts(real)
    synth_probs = normalize_counts(synth)
    keys = set(real_probs).union(synth_probs)
    total_variation = 0.0
    for key in keys:
        total_variation += abs(real_probs.get(key, 0.0) - synth_probs.get(key, 0.0))
    return 0.5 * total_variation


def categorical_trend(real: pd.Series, synth: pd.Series) -> float:
    real_probs = normalize_counts(real)
    synth_probs = normalize_counts(synth)
    if not real_probs:
        return 0.0
    real_mode_prob = max(real_probs.values())
    synth_mode_prob = max(synth_probs.values()) if synth_probs else 0.0
    denom = max(real_mode_prob, 1e-8)
    return abs(real_mode_prob - synth_mode_prob) / denom


def numeric_shape(real: pd.Series, synth: pd.Series) -> float:
    real_clean = real.dropna().to_numpy()
    synth_clean = synth.dropna().to_numpy()
    if real_clean.size == 0 or synth_clean.size == 0:
        return 0.0
    dist = wasserstein_distance(real_clean, synth_clean)
    data_range = real_clean.max() - real_clean.min()
    if data_range <= 1e-12:
        return dist
    return dist / data_range


def numeric_trend(real: pd.Series, synth: pd.Series) -> float:
    real_mean = real.mean()
    synth_mean = synth.mean()
    denom = max(abs(real_mean), 1e-8)
    return abs(real_mean - synth_mean) / denom


def evaluate(real_df: pd.DataFrame, synth_df: pd.DataFrame, cat_cols: List[str], num_cols: List[str], target: Optional[str]) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {
        "categorical": {"shape": 0.0, "trend": 0.0, "count": 0},
        "numeric": {"shape": 0.0, "trend": 0.0, "count": 0},
    }

    for col in cat_cols:
        if col not in real_df.columns or col not in synth_df.columns:
            continue
        shape = categorical_shape(real_df[col], synth_df[col])
        trend = categorical_trend(real_df[col], synth_df[col])
        metrics["categorical"]["shape"] += shape
        metrics["categorical"]["trend"] += trend
        metrics["categorical"]["count"] += 1

    for col in num_cols:
        if col not in real_df.columns or col not in synth_df.columns:
            continue
        shape = numeric_shape(real_df[col], synth_df[col])
        trend = numeric_trend(real_df[col], synth_df[col])
        metrics["numeric"]["shape"] += shape
        metrics["numeric"]["trend"] += trend
        metrics["numeric"]["count"] += 1

    aggregated: Dict[str, Dict[str, float]] = {}
    for kind in ["categorical", "numeric"]:
        count = metrics[kind]["count"]
        if count > 0:
            aggregated[kind] = {
                "shape": metrics[kind]["shape"] / count,
                "trend": metrics[kind]["trend"] / count,
                "columns": count,
            }
        else:
            aggregated[kind] = {"shape": 0.0, "trend": 0.0, "columns": 0}

    total_cols = metrics["categorical"]["count"] + metrics["numeric"]["count"]
    if total_cols > 0:
        aggregated["overall"] = {
            "shape": (
                metrics["categorical"]["shape"] + metrics["numeric"]["shape"]
            ) / total_cols,
            "trend": (
                metrics["categorical"]["trend"] + metrics["numeric"]["trend"]
            ) / total_cols,
            "columns": total_cols,
        }
    else:
        aggregated["overall"] = {"shape": 0.0, "trend": 0.0, "columns": 0}

    if target and target in real_df.columns and target in synth_df.columns:
        if pd.api.types.is_numeric_dtype(real_df[target]) and not pd.api.types.is_bool_dtype(real_df[target]):
            aggregated["target"] = {
                "shape": numeric_shape(real_df[target], synth_df[target]),
                "trend": numeric_trend(real_df[target], synth_df[target]),
                "columns": 1,
            }
        else:
            aggregated["target"] = {
                "shape": categorical_shape(real_df[target], synth_df[target]),
                "trend": categorical_trend(real_df[target], synth_df[target]),
                "columns": 1,
            }

    return aggregated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare real vs synthetic datasets with shape/trend metrics."
    )
    parser.add_argument("--real", type=Path, required=True, help="Path to real CSV file.")
    parser.add_argument("--synthetic", type=Path, required=True, help="Path to synthetic CSV file.")
    parser.add_argument("--info", type=Path, default=None, help="Optional info.json with column metadata.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save metrics JSON.")
    parser.add_argument("--use-wandb", action="store_true", help="Log evaluation metrics to Weights & Biases.")
    parser.add_argument("--wandb-project", type=str, default=None, help="Optional WandB project name.")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Optional WandB run name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    wandb_run = None
    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is requested but not installed. Install it with `pip install wandb`.")
        wandb_run = wandb.init(
            project=args.wandb_project or "tab-sup",
            name=args.wandb_run_name,
            job_type="evaluation",
            config={
                "real_path": str(args.real),
                "synthetic_path": str(args.synthetic),
                "info_path": str(args.info) if args.info else None,
            },
        )

    real_df = pd.read_csv(args.real)
    synth_df = pd.read_csv(args.synthetic)

    info = load_info(args.info)
    cat_cols, num_cols, target = detect_column_types(real_df, info)
    metrics = evaluate(real_df, synth_df, cat_cols, num_cols, target)

    print("=== Shape / Trend metrics ===")
    for section, values in metrics.items():
        print(
            f"{section:>10}: shape={values['shape']:.4f}, trend={values['trend']:.4f}, columns={values['columns']}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2))
        print(f"\nSaved metrics to {args.output}")

    if wandb_run is not None:
        flattened = {}
        for section, values in metrics.items():
            for key, value in values.items():
                flattened[f"{section}/{key}"] = value
        wandb.log(flattened)

        artifact = wandb.Artifact(
            name=f"evaluation-{args.synthetic.stem}-{wandb_run.id}",
            type="evaluation",
            description="Tab-sup synthetic data evaluation inputs and outputs",
        )
        artifact.add_file(str(args.real))
        artifact.add_file(str(args.synthetic))
        if args.output:
            artifact.add_file(str(args.output))
        if args.info and args.info.exists():
            artifact.add_file(str(args.info))
        wandb.log_artifact(artifact)
        wandb_run.finish()


if __name__ == "__main__":
    main()
