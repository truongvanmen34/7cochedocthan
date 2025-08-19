from typing import List
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem, rdMolDescriptors

def smiles_to_mol(smiles: str):
    return Chem.MolFromSmiles(smiles)

def fp_maccs(mol):
    return list(MACCSkeys.GenMACCSKeys(mol))

def fp_ecfp(mol, radius=1, nBits=2048):
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits))

def fp_rdk7(mol, nBits=4096):
    return list(rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=nBits))

def compute_fingerprint_matrix(smiles: List[str], kind: str) -> pd.DataFrame:
    mols = [smiles_to_mol(s) for s in smiles]
    rows = []
    if kind == "maccs":
        rows = [fp_maccs(m) for m in mols]
    elif kind == "ecfp2":
        rows = [fp_ecfp(m, radius=1, nBits=2048) for m in mols]
    elif kind == "ecfp4":
        rows = [fp_ecfp(m, radius=2, nBits=2048) for m in mols]
    elif kind == "rdk7":
        rows = [fp_rdk7(m, nBits=4096) for m in mols]
    else:
        raise ValueError(f"Fingerprint không hỗ trợ: {kind}")
    df = pd.DataFrame(rows)
    df.columns = [f"{kind}_{i}" for i in range(df.shape[1])]
    return df
