#!/usr/bin/env bash
set -e

# Cài thư viện (tương thích Kaggle)
pip install -r requirements.txt

# Chạy thử trên dữ liệu ví dụ (để kiểm tra pipeline)
python -m src.train --data_dir data/example --out_dir outputs/vi_du --fast
