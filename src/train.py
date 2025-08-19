import os, argparse, yaml, json
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from src.data import load_mechanism_csvs, validate_smiles_column, ensure_label
from src.featurize import compute_fingerprint_matrix
from src.models import make_models
from src.metrics import compute_all_metrics
from src.utils import safe_name, set_seed, save_json

def build_argparser():
    ap = argparse.ArgumentParser(description="Huấn luyện phân loại cơ chế độc thận (thuần Việt).")
    ap.add_argument("--data_dir", required=True, help="Thư mục chứa các file CSV (mỗi file một cơ chế: smiles,label).")
    ap.add_argument("--out_dir", required=True, help="Thư mục lưu kết quả.")
    ap.add_argument("--config", default="configs/default.yaml", help="Đường dẫn file YAML cấu hình.")
    ap.add_argument("--fast", action="store_true", help="Chế độ nhanh (MACCS + RandomForest).")
    return ap

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_resampler(kind: str):
    if kind == "none":
        return None
    if kind == "smote":
        return SMOTE(random_state=42)
    if kind == "rus":
        return RandomUnderSampler(random_state=42)
    raise ValueError(f"Kiểu resampling không hỗ trợ: {kind}")

def main():
    args = build_argparser().parse_args()
    cfg = load_config(args.config)

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(cfg.get("seed", 42))

    resampling = cfg.get("resampling", "none")
    fingerprints = cfg.get("fingerprints", ["maccs", "ecfp2", "ecfp4", "rdk7"])
    models_cfg = cfg.get("models", ["svm", "rf", "xgb"])
    grids = cfg["grids"]
    cv_folds = cfg.get("cv_folds", 5)
    test_size = cfg.get("test_size", 0.2)

    if args.fast:
        fingerprints = ["maccs"]
        models_cfg = ["rf"]
        print("[FAST] Chế độ nhanh: fingerprints=MACCS, models=RandomForest, grid nhỏ.")

    csv_files = load_mechanism_csvs(args.data_dir)
    print(f"[INFO] Tìm thấy {len(csv_files)} file cơ chế trong {args.data_dir}")

    summary_rows = []
    best_overall = {}
    partial_csv = os.path.join(args.out_dir, "metrics_summary_partial.csv")

    for csv_path in tqdm(csv_files, desc="Cơ chế", unit="file"):
        mech_name = safe_name(csv_path)
        mech_dir_base = os.path.join(args.out_dir, mech_name)
        os.makedirs(mech_dir_base, exist_ok=True)

        df = pd.read_csv(csv_path)
        df = validate_smiles_column(df, "smiles")
        df = ensure_label(df, "label")

        X_smiles = df["smiles"].tolist()
        y = df["label"].values

        X_train_smi, X_test_smi, y_train, y_test = train_test_split(
            X_smiles, y, test_size=test_size, random_state=cfg.get("seed", 42), stratify=y
        )

        best_mech = None

        for fp_kind in tqdm(fingerprints, desc=f"  FP của {mech_name}", leave=False):
            X_train = compute_fingerprint_matrix(X_train_smi, kind=fp_kind)
            X_test  = compute_fingerprint_matrix(X_test_smi,  kind=fp_kind)

            sampler = get_resampler(resampling)

            # Chuẩn bị model list và grid
            model_specs = make_models(grids)

            for spec in tqdm([m for m in model_specs if m.name in models_cfg], desc=f"    Mô hình", leave=False):
                steps = []
                if sampler is not None:
                    steps.append(("sampler", sampler))
                steps.append(("clf", spec.estimator))

                pipe = ImbPipeline(steps=steps)

                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=cfg.get("seed", 42))
                grid = GridSearchCV(
                    estimator=pipe,
                    param_grid={f"clf__{k}": v for k, v in spec.grid.items()},
                    scoring="f1",
                    cv=cv,
                    n_jobs=-1,
                    verbose=0
                )
                grid.fit(X_train, y_train)

                # Sau CV: nếu là XGB, refit với early stopping trên hold-out từ train
                best_est = grid.best_estimator_
                if spec.name == "xgb":
                    # tách 20% train làm val
                    X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
                        X_train, y_train, test_size=0.2, random_state=cfg.get("seed", 42), stratify=y_train
                    )
                    # refit pipeline với early stopping (chỉ áp dụng cho clf)
                    fit_params = {"clf__eval_set": [(X_val2, y_val2)], "clf__early_stopping_rounds": 20}
                    best_est.fit(X_tr2, y_tr2, **fit_params)

                # Đánh giá trên test
                y_pred = best_est.predict(X_test)
                try:
                    y_proba = best_est.predict_proba(X_test)[:, 1]
                except Exception:
                    y_proba = None

                metrics = compute_all_metrics(y_test, y_pred, y_proba)
                result_row = {
                    "mechanism": mech_name,
                    "fingerprint": fp_kind,
                    "model": spec.name,
                    "best_params": grid.best_params_,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "roc_auc": metrics.get("roc_auc", float("nan"))
                }
                summary_rows.append(result_row)

                # Lưu per-combo
                combo_dir = os.path.join(mech_dir_base, f"{fp_kind}_{spec.name}")
                os.makedirs(combo_dir, exist_ok=True)
                save_json(
                    {"best_params": grid.best_params_, "metrics": metrics},
                    os.path.join(combo_dir, "metrics.json")
                )
                pd.DataFrame(metrics["confusion_matrix"],
                             columns=["pred_0","pred_1"],
                             index=["true_0","true_1"]).to_csv(os.path.join(combo_dir, "confusion_matrix.csv"))

                # Cập nhật best theo (F1, Recall, AUC)
                key = (metrics["f1"], metrics["recall"], metrics.get("roc_auc", float("nan")))
                if (best_mech is None) or (key > best_mech["key"]):
                    best_mech = {
                        "key": key,
                        "fingerprint": fp_kind,
                        "model": spec.name,
                        "best_params": grid.best_params_,
                        "metrics": metrics
                    }

                # Ghi log trung gian
                pd.DataFrame(summary_rows).to_csv(partial_csv, index=False)

        # Lưu best cho cơ chế này
        best_overall[mech_name] = {
            "fingerprint": best_mech["fingerprint"],
            "model": best_mech["model"],
            "best_params": best_mech["best_params"],
            "metrics": best_mech["metrics"]
        }

    # Lưu tổng hợp cuối
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.out_dir, "metrics_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    with open(os.path.join(args.out_dir, "best_models.json"), "w", encoding="utf-8") as f:
        json.dump(best_overall, f, ensure_ascii=False, indent=2)

    print(f"\n[HOÀN TẤT] Tổng hợp: {summary_csv}")
    print(f"[HOÀN TẤT] Best models: {os.path.join(args.out_dir, 'best_models.json')}")

if __name__ == "__main__":
    main()
