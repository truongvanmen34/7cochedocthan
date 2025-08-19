from dataclasses import dataclass
from typing import Dict, Any

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

@dataclass
class ModelSpec:
    name: str
    estimator: Any
    grid: Dict[str, list]

def make_models(grids: dict):
    models = []
    models.append(ModelSpec(
        name="svm",
        estimator=SVC(probability=True, class_weight="balanced"),
        grid={
            "kernel": grids["svm"]["kernel"],
            "C": grids["svm"]["C"],
            "gamma": grids["svm"]["gamma"]
        }
    ))
    models.append(ModelSpec(
        name="rf",
        estimator=RandomForestClassifier(),
        grid={
            "n_estimators": grids["rf"]["n_estimators"],
            "max_depth": grids["rf"]["max_depth"],
            "min_samples_split": grids["rf"]["min_samples_split"],
            "min_samples_leaf": grids["rf"]["min_samples_leaf"],
            "class_weight": grids["rf"]["class_weight"]
        }
    ))
    models.append(ModelSpec(
        name="xgb",
        estimator=XGBClassifier(
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=-1
        ),
        grid={
            "max_depth": grids["xgb"]["max_depth"],
            "learning_rate": grids["xgb"]["learning_rate"],
            "n_estimators": grids["xgb"]["n_estimators"],
            "min_child_weight": grids["xgb"]["min_child_weight"],
            "gamma": grids["xgb"]["gamma"],
            "subsample": grids["xgb"]["subsample"],
            "colsample_bytree": grids["xgb"]["colsample_bytree"]
        }
    ))
    return models
