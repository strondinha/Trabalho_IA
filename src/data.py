from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

NSL_KDD_FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

NSL_KDD_COLUMNS = NSL_KDD_FEATURES + ["label", "difficulty"]
NSL_KDD_COLUMNS_NO_DIFFICULTY = NSL_KDD_FEATURES + ["label"]

ATTACK_CATEGORY_MAP = {
    "normal": "normal",
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos", "smurf": "dos", "teardrop": "dos",
    "mailbomb": "dos", "apache2": "dos", "processtable": "dos", "udpstorm": "dos",
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe", "satan": "probe", "mscan": "probe", "saint": "probe",
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l", "multihop": "r2l", "phf": "r2l", "spy": "r2l",
    "warezclient": "r2l", "warezmaster": "r2l", "sendmail": "r2l", "named": "r2l", "snmpgetattack": "r2l",
    "snmpguess": "r2l", "xlock": "r2l", "xsnoop": "r2l", "worm": "r2l",
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r", "rootkit": "u2r", "httptunnel": "u2r",
    "ps": "u2r", "sqlattack": "u2r", "xterm": "u2r",
}


def load_nsl_kdd_file(file_path: str | Path) -> pd.DataFrame:
    """Load a NSL-KDD split file (KDDTrain+ or KDDTest+) from a CSV-like txt file."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    df = pd.read_csv(
        file_path,
        sep=",",
        header=None,
        skipinitialspace=True,
        skip_blank_lines=True,
    ).dropna(how="all")

    if df.shape[1] == 1:
        raise ValueError(
            f"Falha ao ler '{file_path}': apenas 1 coluna detectada. "
            "Verifique se o arquivo está no formato NSL-KDD (separado por vírgulas)."
        )

    if df.shape[1] == len(NSL_KDD_COLUMNS):
        df.columns = NSL_KDD_COLUMNS
    elif df.shape[1] == len(NSL_KDD_COLUMNS_NO_DIFFICULTY):
        df.columns = NSL_KDD_COLUMNS_NO_DIFFICULTY
    else:
        raise ValueError(
            f"Formato inválido em '{file_path}': {df.shape[1]} colunas encontradas. "
            f"Esperado {len(NSL_KDD_COLUMNS_NO_DIFFICULTY)} (41 features + label) "
            f"ou {len(NSL_KDD_COLUMNS)} (com difficulty)."
        )

    object_cols = df.select_dtypes(include=["object", "string"]).columns
    if len(object_cols) > 0:
        df[object_cols] = df[object_cols].apply(lambda col: col.str.strip())

    df["label"] = df["label"].astype(str).str.strip().str.rstrip(".")
    return df


def load_nsl_kdd_dataset(raw_dir: str | Path = "data/raw", train_file: str = "KDDTrain+.txt", test_file: str = "KDDTest+.txt") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test splits from data/raw/."""
    raw_dir = Path(raw_dir)
    train_df = load_nsl_kdd_file(raw_dir / train_file)
    test_df = load_nsl_kdd_file(raw_dir / test_file)
    return train_df, test_df


def to_attack_category(labels: Iterable[str]) -> pd.Series:
    """Map attack labels to NSL-KDD macro-categories; unknown labels are mapped to 'unknown'."""
    label_series = pd.Series(labels, dtype="string")
    return label_series.str.strip().str.lower().map(ATTACK_CATEGORY_MAP).fillna("unknown")


def split_features_target(df: pd.DataFrame, target_mode: str = "category") -> tuple[pd.DataFrame, pd.Series]:
    """
    target_mode:
      - 'raw': attack names from dataset
      - 'category': mapped macro-categories
      - 'binary': normal vs attack
    """
    X = df[NSL_KDD_FEATURES].copy()
    y_raw = df["label"].astype(str).str.strip().str.rstrip(".").str.lower()

    if target_mode == "raw":
        y = y_raw
    elif target_mode == "category":
        y = to_attack_category(y_raw)
    elif target_mode == "binary":
        y = y_raw.ne("normal").map({True: "attack", False: "normal"})
    else:
        raise ValueError("target_mode deve ser 'raw', 'category' ou 'binary'.")

    return X, y


def infer_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (categorical_columns, numeric_columns)."""
    categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object", "category", "string"]).columns.tolist()
    return categorical_cols, numeric_cols
