### 主要改動說明

1.  **合併輔助函數**: `download_file`, `extract_zip_file`, `extract_ratings_from_reviews` (稍微改名以區分 Amazon) 被移到腳本頂部。
2.  **`process_amazon` 函數**: 封裝了原 `download_amazon.py` 的主要邏輯，接收 `dataset_name` (即 category), `data_version`, `output_dir` 作為參數。
3.  **`process_movielens` 函數**: 封裝了原 `download_movielens.py` 的主要邏輯，接收 `dataset_name`, `output_dir` 作為參數。內部包含了 URL 映射、解壓和格式轉換的邏輯。**注意**: MovieLens 的數據處理部分（格式轉換為 CSV 和 JSON）也被合併進來了，因為它緊跟著下載和解壓。
4.  **`main` 函數**:
    * 使用 `argparse` 接收 `--source`, `--dataset`, `--data_version`, `--output_dir` 參數。
    * 根據 `--source` 的值，調用 `process_amazon` 或 `process_movielens`，並傳遞相應的參數。
    * 為 MovieLens 添加了 `--dataset` 值的驗證。
5.  **路徑結構**:
    * Amazon 數據仍然遵循 `../datasets/amazon{version}/Metadata/`, `../datasets/amazon{version}/Review/`, `../datasets/amazon{version}/Ratings/` 的結構。
    * MovieLens 數據現在會保存在 `../datasets/{dataset_name}/raw/` (原始解壓文件) 和 `../datasets/{dataset_name}/processed/` (處理後的 CSV 和 JSON)。

### 如何使用

```bash
# 下載並處理 Amazon Musical_Instruments (v14)
python download_data.py --source amazon --dataset Baby

# 下載並處理 MovieLens 1M
python download_data.py --source movielens --dataset ml-1m

# 下載並處理 MovieLens 20M
python download_data.py --source movielens --dataset ml-20m
