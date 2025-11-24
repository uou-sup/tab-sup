import json
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib import request
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data")

NAME_URL_DICT_UCI = {
    "adult": "https://archive.ics.uci.edu/static/public/2/adult.zip",
    "default": "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
    "magic": "https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip",
    "shoppers": "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip",
    "beijing": "https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip",
    "news": "https://archive.ics.uci.edu/static/public/332/online+news+popularity.zip",
    "news_nocat": "https://archive.ics.uci.edu/static/public/332/online+news+popularity.zip",
    "diabetes": "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip",
    "adult_dcr": "https://archive.ics.uci.edu/static/public/2/adult.zip",
    "default_dcr": "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
    "magic_dcr": "https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip",
    "shoppers_dcr": "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip",
    "beijing_dcr": "https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip",
    "news_dcr": "https://archive.ics.uci.edu/static/public/332/online+news+popularity.zip",
    "diabetes_dcr": "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip",
}


def unzip_file(zip_filepath: Path, dest_path: Path) -> None:
    with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
        zip_ref.extractall(dest_path)


def download_from_uci(name: str) -> None:
    print(f"Preparing dataset '{name}' from UCI.")
    save_dir = DATA_DIR / name
    save_dir.mkdir(parents=True, exist_ok=True)
    info_path = save_dir / "info.json"

    url = NAME_URL_DICT_UCI[name]
    zip_path = save_dir / f"{name}.zip"
    if not any(save_dir.iterdir()):
        request.urlretrieve(url, zip_path)
        print(f"Downloaded dataset from {url} -> {zip_path}")
        unzip_file(zip_path, save_dir)
        print(f"Unzipped archive for {name}")
    else:
        print(f"Directory {save_dir} already populated, skipping download.")

    if zip_path.exists():
        zip_path.unlink()

    processor = DATASET_PROCESSORS.get(name)
    if processor is None:
        if info_path.exists():
            print(f"Info file already present for {name}.")
        else:
            print(f"No structured processor registered for dataset '{name}'.")
        return

    if info_path.exists():
        print(f"{info_path} already exists. Skipping metadata build.")
        return

    processor(save_dir)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _find_file(root: Path, pattern: str) -> Path:
    matches = list(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not locate file matching '{pattern}' under {root}")
    return matches[0]


def _build_info(
    name: str,
    combined_df: pd.DataFrame,
    target_col: str,
    categorical_cols: List[str],
    task_type: str,
    train_path: Path,
    test_path: Path,
    val_path: Optional[Path],
    train_rows: int,
    test_rows: int,
    val_rows: int,
) -> dict:
    column_names = list(combined_df.columns)
    target_idx = column_names.index(target_col)
    cat_idx = sorted(column_names.index(col) for col in categorical_cols if col in column_names)
    num_idx = [idx for idx in range(len(column_names)) if idx not in cat_idx and idx != target_idx]

    int_idx: List[int] = []
    for idx in num_idx:
        col = column_names[idx]
        series = combined_df[col]
        if pd.api.types.is_integer_dtype(series.dropna()):
            int_idx.append(idx)

    column_info: Dict[str, dict] = {}
    metadata_columns: Dict[str, dict] = {}
    for idx, col in enumerate(column_names):
        entry: Dict[str, object] = {}
        is_numeric = idx in num_idx or (idx == target_idx and task_type == "regression")
        if is_numeric:
            entry["type"] = "numerical"
            series = combined_df[col]
            try:
                entry["min"] = float(np.nanmin(series))
                entry["max"] = float(np.nanmax(series))
            except ValueError:
                entry["min"] = None
                entry["max"] = None
        else:
            entry["type"] = "categorical"
        column_info[str(idx)] = entry
        metadata_columns[str(idx)] = {
            "sdtype": "numerical" if entry["type"] == "numerical" else "categorical",
            "name": col,
        }

    info = {
        "name": name,
        "task_type": task_type,
        "header": "infer",
        "column_names": column_names,
        "num_col_idx": num_idx,
        "cat_col_idx": cat_idx,
        "target_col_idx": [target_idx],
        "file_type": "csv",
        "data_path": str(train_path),
        "val_path": str(val_path) if val_path else None,
        "test_path": str(test_path),
        "int_col_idx": int_idx,
        "int_columns": [column_names[idx] for idx in int_idx],
        "int_col_idx_wrt_num": [num_idx.index(idx) for idx in int_idx],
        "column_info": column_info,
        "train_num": int(train_rows),
        "test_num": int(test_rows),
        "val_num": int(val_rows),
        "idx_mapping": {str(idx): idx for idx in range(len(column_names))},
        "inverse_idx_mapping": {str(idx): idx for idx in range(len(column_names))},
        "idx_name_mapping": {str(idx): column_names[idx] for idx in range(len(column_names))},
        "metadata": {
            "columns": metadata_columns,
            "problem_type": "classification" if task_type != "regression" else "regression",
        },
    }
    return info


def _process_adult(save_dir: Path) -> None:
    columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income",
    ]
    train_file = _find_file(save_dir, "adult.data")
    test_file = _find_file(save_dir, "adult.test")

    train_df = pd.read_csv(train_file, names=columns, skipinitialspace=True)
    test_df = pd.read_csv(test_file, names=columns, skiprows=1, skipinitialspace=True, comment="|")

    def clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).str.strip().replace("?", "Unknown")
        df["income"] = (
            df["income"].astype(str).str.replace(".", "", regex=False).str.strip()
        )
        df["income"] = df["income"].map({">50K": 1, "<=50K": 0})
        numeric_cols = [
            "age",
            "fnlwgt",
            "education_num",
            "capital_gain",
            "capital_loss",
            "hours_per_week",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    train_df = clean(train_df)
    test_df = clean(test_df)

    train_path = save_dir / "train.csv"
    test_path = save_dir / "test.csv"
    _save_csv(train_df, train_path)
    _save_csv(test_df, test_path)

    combined = pd.concat([train_df, test_df], ignore_index=True)
    info = _build_info(
        name="adult",
        combined_df=combined,
        target_col="income",
        categorical_cols=[
            "workclass",
            "education",
            "marital_status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native_country",
        ],
        task_type="binclass",
        train_path=train_path,
        test_path=test_path,
        val_path=None,
        train_rows=len(train_df),
        test_rows=len(test_df),
        val_rows=0,
    )
    (save_dir / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"Prepared structured files for adult at {save_dir}")


def _split_and_write(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    save_dir: Path,
    dataset_name: str,
    categorical_cols: List[str],
    test_size: float = 0.2,
    random_state: int = 777,
) -> None:
    stratify = df[target_col] if task_type != "regression" else None
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify
    )
    train_path = save_dir / "train.csv"
    test_path = save_dir / "test.csv"
    _save_csv(train_df, train_path)
    _save_csv(test_df, test_path)

    combined = pd.concat([train_df, test_df], ignore_index=True)
    info = _build_info(
        name=dataset_name,
        combined_df=combined,
        target_col=target_col,
        categorical_cols=categorical_cols,
        task_type=task_type,
        train_path=train_path,
        test_path=test_path,
        val_path=None,
        train_rows=len(train_df),
        test_rows=len(test_df),
        val_rows=0,
    )
    (save_dir / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"Prepared structured files for {dataset_name} at {save_dir}")


def _process_default(save_dir: Path) -> None:
    excel_path = _find_file(save_dir, "*.xls")
    try:
        df = pd.read_excel(excel_path, header=1)
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Reading the credit default Excel file requires the 'xlrd' package. "
            "Install it with `pip install xlrd` and rerun."
        ) from exc

    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    target_col = "default_payment_next_month"
    df[target_col] = df[target_col].astype(int)

    categorical_cols = ["sex", "education", "marriage"] + [col for col in df.columns if col.startswith("pay_")]

    _split_and_write(
        df=df,
        target_col=target_col,
        task_type="binclass",
        save_dir=save_dir,
        dataset_name="default",
        categorical_cols=categorical_cols,
    )


def _process_magic(save_dir: Path) -> None:
    columns = [f"f{i}" for i in range(1, 11)] + ["class_label"]
    data_path = _find_file(save_dir, "magic04.data")
    df = pd.read_csv(data_path, names=columns, skipinitialspace=True)
    df["class_label"] = df["class_label"].str.strip().map({"g": 1, "h": 0})

    _split_and_write(
        df=df,
        target_col="class_label",
        task_type="binclass",
        save_dir=save_dir,
        dataset_name="magic",
        categorical_cols=[],
    )


def _process_beijing(save_dir: Path) -> None:
    csv_path = _find_file(save_dir, "PRSA_data_2010.1.1-2014.12.31.csv")
    df = pd.read_csv(csv_path)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    if "no" in df.columns:
        df = df.drop(columns=["no"])
    if "pm2.5" in df.columns:
        df = df.rename(columns={"pm2.5": "pm2_5"})
    df = df.dropna(subset=["pm2_5"])

    categorical_cols = ["cbwd"] if "cbwd" in df.columns else []

    _split_and_write(
        df=df,
        target_col="pm2_5",
        task_type="regression",
        save_dir=save_dir,
        dataset_name="beijing",
        categorical_cols=categorical_cols,
    )


def _process_news(save_dir: Path) -> None:
    csv_path = _find_file(save_dir, "OnlineNewsPopularity.csv")
    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]
    if "url" in df.columns:
        df = df.drop(columns=["url"])
    df = df.rename(columns=lambda col: col.replace(" ", "_"))

    _split_and_write(
        df=df,
        target_col="shares",
        task_type="regression",
        save_dir=save_dir,
        dataset_name="news",
        categorical_cols=[],
    )


DATASET_PROCESSORS: Dict[str, Callable[[Path], None]] = {
    "adult": _process_adult,
    "default": _process_default,
    "magic": _process_magic,
    "beijing": _process_beijing,
    "news": _process_news,
}


if __name__ == "__main__":
    for dataset_name in NAME_URL_DICT_UCI:
        download_from_uci(dataset_name)
