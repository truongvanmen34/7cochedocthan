import os, re, json, random, pathlib
import numpy as np

def safe_name(pathlike: str) -> str:
    base = os.path.basename(pathlike)
    name = re.sub(r'\.[^.]+$', '', base)
    name = re.sub(r'[^A-Za-z0-9._-]+', '_', name)
    return name

def set_seed(seed: int = 42):
    import numpy as np, random
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    random.seed(seed)
    np.random.seed(seed)

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
