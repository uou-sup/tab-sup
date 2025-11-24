import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import utils


def _resolve_path(path_str: Optional[str]) -> Optional[Path]:
    if path_str is None:
        return None
    path = Path(path_str)
    if path.exists():
        return path
    return Path.cwd() / path


def _save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=array.dtype == object)


def _extract_columns(info: dict, indices: List[int]) -> List[str]:
    column_names = info.get("column_names")
    if not column_names:
        raise ValueError("info.json must include column_names.")
    return [column_names[idx] for idx in indices]


def _encode_targets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    task_type: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[List[str]], Optional[int]]:
    if task_type.lower() == "regression":
        return (
            train_df[target_col].astype(np.float32, copy=False).to_numpy(),
            val_df[target_col].astype(np.float32, copy=False).to_numpy(),
            test_df[target_col].astype(np.float32, copy=False).to_numpy(),
            None,
            None,
        )

    combined = pd.concat(
        [train_df[target_col], val_df[target_col], test_df[target_col]], axis=0, ignore_index=True
    )
    categories = pd.Categorical(combined)
    target_classes = [str(cat) for cat in categories.categories]
    mapping = {cat: idx for idx, cat in enumerate(categories.categories)}

    def _map(df: pd.DataFrame) -> np.ndarray:
        mapped = df[target_col].map(mapping)
        if mapped.isna().any():
            missing = df.loc[mapped.isna(), target_col].unique()
            raise ValueError(f"Unknown target labels encountered: {missing}")
        return mapped.astype(np.float32, copy=False).to_numpy()

    return _map(train_df), _map(val_df), _map(test_df), target_classes, len(target_classes)


def _make_feature_arrays(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    if numeric_cols:
        X_num = df[numeric_cols].to_numpy(dtype=np.float32, copy=False)
    else:
        X_num = np.empty((len(df), 0), dtype=np.float32)

    if categorical_cols:
        X_cat = df[categorical_cols].astype(object).to_numpy()
    else:
        X_cat = np.empty((len(df), 0), dtype=object)

    return X_num, X_cat


def prepare_dataset_from_info(
    dataset_name: str,
    *,
    val_ratio: float = 0.1,
    random_state: int = 777,
) -> None:
    data_dir = Path("data") / dataset_name
    info_path = data_dir / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing metadata: {info_path}")
    info = utils.load_json(info_path)

    train_path = _resolve_path(info.get("data_path"))
    if train_path is None or not train_path.exists():
        raise FileNotFoundError(f"Training CSV not found (expected {info.get('data_path')}).")
    train_df = pd.read_csv(train_path)

    test_path = _resolve_path(info.get("test_path"))
    if test_path is not None and test_path.exists():
        test_df = pd.read_csv(test_path)
    else:
        test_df = pd.DataFrame(columns=train_df.columns)

    val_path = _resolve_path(info.get("val_path"))
    if val_path is not None and val_path.exists():
        val_df = pd.read_csv(val_path)
    else:
        val_df = pd.DataFrame(columns=train_df.columns)

    task_type = info.get("task_type", "binclass").lower()
    target_idx = info.get("target_col_idx")
    if not target_idx:
        raise ValueError("info.json must include target_col_idx.")
    column_names = info.get("column_names")
    if column_names is None:
        raise ValueError("info.json must include column_names.")
    target_col = column_names[target_idx[0]]

    if val_df.empty and val_ratio > 0:
        stratify = train_df[target_col] if task_type != "regression" else None
        train_df, val_df = train_test_split(
            train_df,
            test_size=val_ratio,
            random_state=random_state,
            stratify=stratify,
        )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    numeric_cols = _extract_columns(info, info.get("num_col_idx", []))
    categorical_cols = _extract_columns(info, info.get("cat_col_idx", []))

    y_train, y_val, y_test, target_classes, n_classes = _encode_targets(
        train_df, val_df, test_df, target_col, task_type
    )
    X_num_train, X_cat_train = _make_feature_arrays(train_df, numeric_cols, categorical_cols)
    X_num_val, X_cat_val = _make_feature_arrays(val_df, numeric_cols, categorical_cols)
    X_num_test, X_cat_test = _make_feature_arrays(test_df, numeric_cols, categorical_cols)

    output_dir = Path("dataset") / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_array(output_dir / "y_train.npy", y_train.astype(np.float32, copy=False))
    _save_array(output_dir / "y_val.npy", y_val.astype(np.float32, copy=False))
    _save_array(output_dir / "y_test.npy", y_test.astype(np.float32, copy=False))

    if numeric_cols:
        _save_array(output_dir / "X_num_train.npy", X_num_train)
        _save_array(output_dir / "X_num_val.npy", X_num_val)
        _save_array(output_dir / "X_num_test.npy", X_num_test)
    else:
        for split, arr in [
            ("train", X_num_train),
            ("val", X_num_val),
            ("test", X_num_test),
        ]:
            _save_array(output_dir / f"X_num_{split}.npy", arr)

    if categorical_cols:
        _save_array(output_dir / "X_cat_train.npy", X_cat_train)
        _save_array(output_dir / "X_cat_val.npy", X_cat_val)
        _save_array(output_dir / "X_cat_test.npy", X_cat_test)
    else:
        for split, arr in [
            ("train", X_cat_train),
            ("val", X_cat_val),
            ("test", X_cat_test),
        ]:
            _save_array(output_dir / f"X_cat_{split}.npy", arr)

    cat_group_sizes = [
        int(train_df[col].nunique(dropna=False)) for col in categorical_cols
    ]

    total_size = len(train_df) + len(val_df) + len(test_df)
    dataset_info = {
        "task_type": task_type,
        "n_classes": n_classes,
        "n_num_features": len(numeric_cols),
        "n_cat_features": len(categorical_cols),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "cat_group_sizes": cat_group_sizes,
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
        "target": target_col,
        "target_classes": target_classes,
        "random_state": random_state,
        "val_ratio": len(val_df) / total_size if total_size else 0.0,
        "test_ratio": len(test_df) / total_size if total_size else 0.0,
        "name": dataset_name,
    }

    info_out_path = output_dir / "info.json"
    info_out_path.write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2))
    print(f"Saved processed dataset to {output_dir}")
