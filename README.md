# Phân loại **cơ chế độc thận** (Model 2) — Bản Thuần Việt, tối ưu cho Kaggle

Repo gọn – dễ dùng – tránh timeout. Chỉ cần có các file **CSV** (mỗi cơ chế 1 file: `smiles,label`) là huấn luyện được ngay.

---

## 🚀 Cách dùng nhanh (Kaggle / Local)

### 1) Chuẩn bị dữ liệu
Tạo thư mục trong `data/` và đặt ~7 file CSV (mỗi file là **1 cơ chế**):
```
data/kidney_mechs/
  necrosis.csv
  fibrosis.csv
  ...
```
Mỗi file có **2 cột**:
```csv
smiles,label
CCO,0
CC(=O)O,1
```
> `label`: 0 = không gây độc theo **cơ chế** đó; 1 = gây độc theo **cơ chế** đó.

Bạn có thể chạy thử với ví dụ: `data/example/example.csv`.

### 2) Cài thư viện
```bash
pip install -r requirements.txt
```

### 3) Huấn luyện
- **Đầy đủ** (4 fingerprint × 3 mô hình × CV 5-fold):
```bash
python -m src.train --data_dir data/kidney_mechs --out_dir outputs/lan1
```
- **Nhanh** (test pipeline, tránh timeout — MACCS + RandomForest):
```bash
python -m src.train --data_dir data/kidney_mechs --out_dir outputs/lan1 --fast
```

### 4) Kết quả
- Tổng hợp: `outputs/lan1/metrics_summary.csv`
- Mô hình tốt nhất mỗi cơ chế: `outputs/lan1/best_models.json`
- Từng cơ chế có thư mục con: `metrics.json` (tham số + chỉ số), `confusion_matrix.csv`

> Trong khi chạy, repo hiển thị **tiến trình** (tqdm) và **ghi log trung gian**:
> `outputs/lan1/metrics_summary_partial.csv` để không mất dữ liệu nếu kaggle dừng giữa chừng.

---

## ⚙️ Mặc định & Tối ưu

- **Fingerprints (RDKit)**: `maccs (166)`, `ecfp2 (2048)`, `ecfp4 (2048)`, `rdk7 (4096)`
- **Mô hình**: `svm`, `rf`, `xgb`
- **Chia dữ liệu**: stratified 80/20 (train/test)
- **GridSearchCV**: 5-fold, `scoring=f1`, `n_jobs=-1`
- **XGBoost**: sau khi tìm best-params bằng CV, **refit lại với early stopping** trên 20% train (hold-out)
- **Mất cân bằng**: chọn trong `configs/default.yaml` → `none | smote | rus` (chỉ áp dụng trên train)
- **Chọn best**: ưu tiên `F1 → Recall → ROC-AUC`

> Repo **không cần Java** (không dùng PaDEL) → chạy mượt trên Kaggle. Nếu cần thêm fingerprint PaDEL,
> hãy mở issue/PR hoặc mở rộng `src/featurize.py` theo hướng dẫn.

---

## 📂 Cấu trúc
```
.
├── configs/
│   └── default.yaml        # cấu hình (grid nhỏ, hợp lý tránh timeout)
├── data/
│   └── example/example.csv # dữ liệu mẫu
├── outputs/                # kết quả
├── src/
│   ├── data.py             # đọc & kiểm tra CSV
│   ├── featurize.py        # sinh fingerprint RDKit
│   ├── models.py           # mô hình + grid
│   ├── metrics.py          # tính chỉ số
│   ├── utils.py            # seed, lưu JSON, tên cơ chế
│   └── train.py            # pipeline chính + tiến trình + fast mode + early stopping (XGB)
├── requirements.txt        # thư viện
├── run.sh                  # chạy thử nhanh
└── README.md               # hướng dẫn này (thuần Việt)
```

---

## 🧪 Chạy thử nhanh
```bash
bash run.sh
```
Sẽ cài thư viện và train trên `data/example/example.csv` để kiểm tra pipeline.
