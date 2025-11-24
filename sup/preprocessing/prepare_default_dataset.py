import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split

from .prepare_dataset_from_info import prepare_dataset_from_info


def _build_info_dict(
    df: pd.DataFrame,
    dataset_name: str,
    target_col: str,
    categorical_cols: List[str],
    train_path: Path,
    test_path: Path,
) -> Dict:
    column_names = list(df.columns)
    target_idx = column_names.index(target_col)
    cat_idx = [column_names.index(col) for col in categorical_cols]
    num_idx = [idx for idx in range(len(column_names)) if idx not in cat_idx and idx != target_idx]

    int_idx = []
    for idx in num_idx:
        col = column_names[idx]
        if pd.api.types.is_integer_dtype(df[col]):
            int_idx.append(idx)

    column_info: Dict[str, Dict] = {}
    metadata_columns: Dict[str, Dict] = {}
    for idx, col in enumerate(column_names):
        entry: Dict[str, object] = {}
        is_numeric = idx in num_idx or idx == target_idx and df[col].dtype.kind in "if"
        if is_numeric:
            entry["type"] = "numerical"
            entry["min"] = float(df[col].min())
            entry["max"] = float(df[col].max())
        else:
            entry["type"] = "categorical"
        column_info[str(idx)] = entry
        metadata_columns[str(idx)] = {
            "sdtype": "numerical" if entry["type"] == "numerical" else "categorical",
            "name": col,
        }

    info = {
        "name": dataset_name,
        "task_type": "binclass",
        "header": "infer",
        "column_names": column_names,
        "num_col_idx": num_idx,
        "cat_col_idx": cat_idx,
        "target_col_idx": [target_idx],
        "file_type": "csv",
        "data_path": str(train_path),
        "val_path": None,
        "test_path": str(test_path),
        "int_col_idx": int_idx,
        "int_columns": [column_names[idx] for idx in int_idx],
        "int_col_idx_wrt_num": [num_idx.index(idx) for idx in int_idx],
        "column_info": column_info,
        "train_num": 0,
        "test_num": 0,
        "val_num": 0,
        "idx_mapping": {str(idx): idx for idx in range(len(column_names))},
        "inverse_idx_mapping": {str(idx): idx for idx in range(len(column_names))},
        "idx_name_mapping": {str(idx): column_names[idx] for idx in range(len(column_names))},
        "metadata": {
            "columns": metadata_columns,
            "problem_type": "classification",
        },
    }
    return info


def _ensure_default_metadata(random_state: int) -> None:
    data_dir = Path("data/default")
    data_dir.mkdir(parents=True, exist_ok=True)
    info_path = data_dir / "info.json"
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    if info_path.exists() and train_csv.exists() and test_csv.exists():
        return

    excel_files = list(data_dir.glob("*.xls"))
    if not excel_files:
        raise FileNotFoundError("Default dataset Excel file not found in data/default.")
    df = pd.read_excel(excel_files[0], header=1)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    target_col = "default_payment_next_month"
    if target_col not in df.columns:
        raise ValueError(f"Expected target column '{target_col}' in default dataset.")
    df[target_col] = df[target_col].astype(int)
    categorical_cols = ["sex", "education", "marriage"] + [col for col in df.columns if col.startswith("pay_")]

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=random_state, stratify=df[target_col]
    )
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    info = _build_info_dict(df, "default", target_col, categorical_cols, train_csv, test_csv)
    info["train_num"] = len(train_df)
    info["test_num"] = len(test_df)
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"Wrote default metadata to {info_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Default Credit dataset tensors.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio if val split absent.")
    parser.add_argument("--random-state", type=int, default=777, help="Random seed for splitting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_default_metadata(args.random_state)
    prepare_dataset_from_info(
        "default",
        val_ratio=args.val_ratio,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
