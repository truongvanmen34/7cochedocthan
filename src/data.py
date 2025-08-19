import os
import pandas as pd
from rdkit import Chem

def load_mechanism_csvs(data_dir: str):
    csvs = []
    for fn in os.listdir(data_dir):
        if fn.lower().endswith(".csv"):
            csvs.append(os.path.join(data_dir, fn))
    if not csvs:
        raise FileNotFoundError(f"Không tìm thấy file CSV nào trong thư mục: {data_dir}")
    return sorted(csvs)

def validate_smiles_column(df: pd.DataFrame, smiles_col: str = "smiles"):
    if smiles_col not in df.columns:
        raise ValueError(f"Thiếu cột '{smiles_col}'. Cột hiện có: {df.columns.tolist()}")
    mask = df[smiles_col].astype(str).apply(lambda s: Chem.MolFromSmiles(s) is not None)
    return df.loc[mask].reset_index(drop=True)

def ensure_label(df: pd.DataFrame, label_col: str = "label"):
    if label_col not in df.columns:
        raise ValueError(f"Thiếu cột '{label_col}'.")
    df[label_col] = df[label_col].astype(int)
    return df
