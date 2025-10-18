# 檔案路徑: build_instruction.py (修正版，處理字串 Item ID)

import json
import os
import argparse
from tqdm import tqdm
from pathlib import Path # 使用 pathlib 處理路徑更佳

def build_instruction_data(codebook_path, source_data_path, output_path, instruction_text, history_prefix, item_separator="||"):
    """
    從 codebook 和源數據構建指令微調數據。
    【已修正】能夠處理源數據中 history 和 target 為字串列表/字串的情況。
    """
    print("-" * 30)
    print(f"Processing source: {source_data_path}")
    
    # --- 檢查輸入文件是否存在 ---
    if not codebook_path.exists():
        print(f"錯誤：Codebook 檔案未找到！ {codebook_path}")
        return False 
    if not source_data_path.exists():
        print(f"警告：源數據檔案未找到，跳過此分割： {source_data_path}")
        return False 
        
    print(f"讀取 Codebook 映射從: {codebook_path}")
    try:
        with open(codebook_path, 'r', encoding='utf-8') as f:
            item_id_to_code_str_map = {int(k): v for k, v in json.load(f).items()}
    except Exception as e:
        print(f"讀取或解析 Codebook 時出錯: {e}")
        return False
    print(f"Codebook 讀取完成，包含 {len(item_id_to_code_str_map)} 個項目。")
    
    print(f"讀取源數據從: {source_data_path}")
    print(f"開始構建指令數據，將輸出到: {output_path}")
    
    # 確保輸出目錄存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    missing_history_items = 0
    missing_target_items = 0
    
    try:
        with open(source_data_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
             
            lines = infile.readlines()
            total_lines = len(lines)
            
            for line in tqdm(lines, total=total_lines, desc=f"Processing {source_data_path.name}"):
                line = line.strip()
                if not line: continue
                    
                try:
                    data = json.loads(line)
                    # ✅ 關鍵修正：將 history 中的字串轉為整數
                    history_ids_str = data.get("history", []) 
                    history_ids = [int(x) for x in history_ids_str] # 0-based integers
                    
                    # ✅ 關鍵修正：將 target 字串轉為整數
                    target_id_str = data.get("target")        
                    if target_id_str is None: continue # 跳過沒有 target 的行
                    target_id = int(target_id_str) # 0-based integer

                    # --- 轉換 History ---
                    history_code_strings = []
                    current_missing = 0
                    for item_id in history_ids: # 現在 history_ids 是整數列表
                        code_str = item_id_to_code_str_map.get(item_id) 
                        if code_str:
                            history_code_strings.append(code_str)
                        else:
                            current_missing += 1
                    missing_history_items += current_missing

                    # --- 轉換 Target ---
                    target_code_str = item_id_to_code_str_map.get(target_id) # target_id 現在是整數
                    if not target_code_str:
                        missing_target_items += 1
                        continue 

                    # --- 格式化 Input 和 Output ---
                    formatted_history = f" {item_separator} ".join(history_code_strings)
                    input_str = f"{history_prefix}\n[{formatted_history}]."
                    output_str = target_code_str 

                    # --- 構建 JSON 物件 ---
                    instruction_json = {
                        "instruction": instruction_text,
                        "input": input_str,
                        "output": output_str
                    }
                    
                    outfile.write(json.dumps(instruction_json, ensure_ascii=False) + '\n')
                    count += 1
                    
                except json.JSONDecodeError:
                    print(f"警告：無法解析行: {line}")
                except ValueError as ve: # 捕捉可能的 int() 轉換錯誤
                    print(f"警告：轉換 Item ID 為整數時出錯 (跳過此行): {line} - {ve}")
                except Exception as e:
                    print(f"處理行 {count+1} 時發生錯誤: {e}")

        print("-" * 30)
        print(f"指令數據構建完成 ({source_data_path.name})！")
        print(f"成功處理並寫入 {count} 條數據到 {output_path}")
        print(f"遇到的缺失 History Item ID 總數: {missing_history_items}")
        print(f"因缺失 Target Item ID 而跳過的數據行數: {missing_target_items}")
        print("-" * 30)
        return True 

    except Exception as e:
        print(f"處理檔案時發生錯誤 ({source_data_path.name}): {e}")
        return False


# --- 主函數入口 (保持不變) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自動處理所有數據分割，構建指令微調數據")
    
    # --- 核心參數 ---
    parser.add_argument('--dataset_name', type=str, required=True, help='數據集名稱')
    parser.add_argument('--quant_method', type=str, required=True, help='量化方法')
    
    # --- 路徑參數 ---
    parser.add_argument('--data_base_path', type=str, default='../datasets', help='數據集根目錄')
    parser.add_argument('--codebook_dir_name', type=str, default='codebooks', help='Codebook JSON 子目錄')
    
    # --- 文本模板參數 ---
    parser.add_argument('--instruction', type=str, default="You are a helpful recommendation assistant. Predict the next item code sequence based on the user's purchase history sequence.", help='指令文本')
    parser.add_argument('--history_prefix', type=str, default="Given the following purchase history of a user (items separated by ||):", help='歷史記錄前綴')
    parser.add_argument('--item_separator', type=str, default="||", help='項目分隔符')

    args = parser.parse_args()

    # --- 自動構建 Codebook 路徑 ---
    dataset_path = Path(args.data_base_path) / args.dataset_name
    codebook_filename = f"{args.dataset_name}.{args.quant_method}.codebook.json"
    codebook_path = dataset_path / args.codebook_dir_name / codebook_filename
    
    # --- 遍歷所有數據分割 ---
    splits_to_process = ['train', 'valid', 'test']
    all_successful = True

    print(f"\n===== 開始處理數據集: {args.dataset_name}, 量化方法: {args.quant_method} =====")
    
    for split in splits_to_process:
        source_filename = f"{args.dataset_name}.{split}.jsonl"
        source_data_path = dataset_path / source_filename
        
        output_dir = dataset_path / "prompts"
        output_filename = f"{args.dataset_name}.{args.quant_method}.{split}.instruction.jsonl"
        output_path = output_dir / output_filename

        success = build_instruction_data(
            codebook_path, source_data_path, output_path,
            args.instruction, args.history_prefix, args.item_separator
        )
        if not success: all_successful = False 

    print("=" * 60)
    if all_successful: print("✅ 所有數據分割處理完成！")
    else: print("⚠️ 部分數據分割處理失敗或被跳過，請檢查上面的日誌。")
    print(f"輸出檔案位於: {dataset_path / 'prompts'}")
    print("=" * 60)