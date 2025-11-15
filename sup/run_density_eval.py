"""
Utility CLI to run the density/shape/trend evaluation used for shoppers.

Example:
python -m sup.run_density_eval --real data/shoppers/online_shoppers_intention.csv \
    --synthetic samples/shoppers_yaml_test.csv \
    --info dataset/shoppers/info.json \
    --per-column-output eval/custom/per_column_shape_trend.csv
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

try:
    from .metrics import TabMetrics
except ImportError:  # pragma: no cover - support `python sup/run_density_eval.py`
    from sup.metrics import TabMetrics  # type: ignore

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    wandb = None  # type: ignore


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run density/shape/trend evaluation between real and synthetic CSVs.")
    parser.add_argument("--real", type=Path, required=True, help="Path to the original (real) dataset CSV.")
    parser.add_argument("--synthetic", type=Path, required=True, help="Path to the generated synthetic CSV.")
    parser.add_argument("--info", type=Path, required=True, help="Path to the dataset info.json metadata.")
    parser.add_argument(
        "--per-column-output",
        type=Path,
        default=None,
        help="Optional path to save per-column shape/trend scores (CSV).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device to run the metric on (default: cpu).",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Log metrics/tables to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Optional WandB project (defaults to tab-sup-eval).",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional WandB run name.",
    )
    return parser.parse_args()


def _load_metadata(info_path: Path) -> dict:
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _align_columns(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in real_df.columns if col not in synthetic_df.columns]
    if missing:
        raise ValueError(f"Synthetic data is missing columns: {missing}")
    return synthetic_df[real_df.columns]


def _compute_per_column_tables(extras: dict, columns: pd.Index) -> pd.DataFrame:
    shape_df = extras["shapes"].copy()
    if pd.api.types.is_integer_dtype(shape_df["Column"]):
        shape_df["Column"] = shape_df["Column"].map(lambda i: columns[i])
    shape_df = shape_df.rename(columns={"Column": "col_name", "Score": "shape_score"})
    shape_df = shape_df[["col_name", "shape_score"]]

    trend_df = extras["trends"].copy()
    if pd.api.types.is_integer_dtype(trend_df["Column 1"]):
        trend_df["Column 1"] = trend_df["Column 1"].map(lambda i: columns[i])
    if pd.api.types.is_integer_dtype(trend_df["Column 2"]):
        trend_df["Column 2"] = trend_df["Column 2"].map(lambda i: columns[i])
    trend_df = trend_df.rename(columns={"Column 1": "col1", "Column 2": "col2", "Score": "trend_score"})

    trend_per_col = {}
    for col in columns:
        related_pairs = trend_df[(trend_df["col1"] == col) | (trend_df["col2"] == col)]
        trend_per_col[col] = related_pairs["trend_score"].mean() if not related_pairs.empty else None
    trend_df_final = pd.DataFrame.from_dict(trend_per_col, orient="index", columns=["trend_score"]).reset_index()
    trend_df_final = trend_df_final.rename(columns={"index": "col_name"})

    shape_df["col_name"] = shape_df["col_name"].astype(str)
    trend_df_final["col_name"] = trend_df_final["col_name"].astype(str)
    merged = pd.merge(shape_df, trend_df_final, on="col_name", how="outer")
    return merged.sort_values(by="col_name").reset_index(drop=True)


def _save_per_column(df: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)


def _log_to_wandb(run, metrics: dict, per_column: pd.DataFrame, extras: dict, args: argparse.Namespace) -> None:
    assert wandb is not None
    flattened = {f"density_eval/{k}": v for k, v in metrics.items()}
    run.summary.update(flattened)

    payload = dict(flattened)
    payload["density_eval/per_column_table"] = wandb.Table(dataframe=per_column)
    payload["density_eval/shape_details_table"] = wandb.Table(dataframe=extras["shapes"])
    payload["density_eval/trend_details_table"] = wandb.Table(dataframe=extras["trends"])
    payload["density_eval/real_path"] = str(args.real)
    payload["density_eval/synthetic_path"] = str(args.synthetic)
    if args.per_column_output is not None:
        payload["density_eval/per_column_csv_path"] = str(args.per_column_output)
    wandb.log(payload)


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = _parse_args()

    wandb_run = None
    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is requested but not installed. Install it with `pip install wandb`.")
        wandb_run = wandb.init(
            project=args.wandb_project or "tab-sup-eval",
            name=args.wandb_run_name,
            job_type="density-eval",
            config={
                "real_csv": str(args.real),
                "synthetic_csv": str(args.synthetic),
                "info_json": str(args.info),
                "device": args.device,
                "metric_list": ["density"],
            },
        )

    info = _load_metadata(args.info)
    real_df = pd.read_csv(args.real)
    syn_df = pd.read_csv(args.synthetic)
    syn_df = _align_columns(real_df, syn_df)

    metric = TabMetrics(
        real_data_path=str(args.real),
        test_data_path=None,
        val_data_path=None,
        info=info,
        device=torch.device(args.device),
        metric_list=["density"],
    )
    metrics, extras = metric.evaluate(syn_df)

    per_column = _compute_per_column_tables(extras, syn_df.columns)
    print("==== Per-column Shape & Trend Results ====")
    print(per_column)
    if args.per_column_output is not None:
        _save_per_column(per_column, args.per_column_output)
        print(f"Per-column scores saved to {args.per_column_output}")

    print("==== Density Evaluation Results ====")
    print(json.dumps(metrics, indent=4))
    print("\n==== Shape Details (first rows) ====")
    print(extras["shapes"].head())
    print("\n==== Trend Details (first rows) ====")
    print(extras["trends"].head())

    if wandb_run is not None:
        _log_to_wandb(wandb_run, metrics, per_column, extras, args)
        wandb_run.finish()


if __name__ == "__main__":
    main()
