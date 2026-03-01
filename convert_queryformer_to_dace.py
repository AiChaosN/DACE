import os
import json
import glob
import pandas as pd
from tqdm import tqdm

# 配置路径
ROOT_DIR = "/home/AiChaosN/Project/Workspace/01_Research/DACE"
SOURCE_DIR = os.path.join(ROOT_DIR, "data/queryformer_data")
TARGET_DIR = os.path.join(ROOT_DIR, "data/queryformer_dace")

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def clean_node(node):
    """
    Extracts only the fields required by DACE from a node dict.
    Recurses into 'Plans'.
    """
    # queryformer_data 的 json 已经是 Postgres 格式的键名，
    # 我们只需要提取 DACE 关心的字段。
    
    dace_node = {
        "Node Type": node.get("Node Type", "Unknown"),
        "Total Cost": node.get("Total Cost", 0.0),
        "Plan Rows": node.get("Plan Rows", 0.0),
        "Actual Total Time": node.get("Actual Total Time", 0.0),
        "Startup Cost": node.get("Startup Cost", 0.0)
    }
    
    # 递归处理子节点
    children = node.get("Plans", [])
    if children:
        dace_node["Plans"] = [clean_node(child) for child in children]
        
    return dace_node

def convert_csvs_to_json(file_pattern, output_filename):
    """
    读取匹配 file_pattern 的所有 CSV 文件，
    解析其中的 'json' 列，转换为 DACE 格式，
    并保存到 output_filename。
    """
    search_path = os.path.join(SOURCE_DIR, file_pattern)
    files = glob.glob(search_path)
    files.sort() # 确保处理顺序一致
    
    if not files:
        print(f"Warning: No files found for pattern: {search_path}")
        return

    print(f"Converting {len(files)} files for {output_filename}...")
    
    all_plans = []
    
    for file_path in tqdm(files, desc=output_filename):
        try:
            # 使用 pandas 读取 CSV，它可以很好地处理引号转义
            df = pd.read_csv(file_path)
            
            if 'json' not in df.columns:
                print(f"Warning: 'json' column not found in {file_path}")
                continue
                
            for _, row in df.iterrows():
                json_str = row['json']
                try:
                    plan_data = json.loads(json_str)
                    
                    # CSV 中的结构通常是 {"Plan": {...root node...}, ...}
                    # 我们需要提取 "Plan" 对应的值
                    root_node = None
                    if "Plan" in plan_data:
                        root_node = plan_data["Plan"]
                    elif "Node Type" in plan_data:
                        # 已经是根节点的情况
                        root_node = plan_data
                    
                    if root_node:
                        cleaned_tree = clean_node(root_node)
                        
                        # DACE setup.py 期望的结构: [{"Plan": ...}]
                        wrapped_plan = {"Plan": cleaned_tree}
                        all_plans.append(wrapped_plan)
                    else:
                        print(f"Warning: No valid plan found in row in {file_path}")
                            
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in {file_path}: {e}")
                    
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    output_path = os.path.join(TARGET_DIR, output_filename)
    print(f"Saving {len(all_plans)} plans to {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(all_plans, f)

def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory {SOURCE_DIR} does not exist.")
        return

    ensure_dir(TARGET_DIR)
    
    # 1. 处理 job-light
    convert_csvs_to_json("job-light_plan.csv", "job_light.json")
    
    # 2. 处理 synthetic
    convert_csvs_to_json("synthetic_plan.csv", "synthetic.json")
    
    # 3. 处理 train parts (train_plan_part0.csv 到 train_plan_part19.csv)
    # 合并为一个大的 train.json
    convert_csvs_to_json("train_plan_part*.csv", "train.json")

    print("\nAll conversions completed.")
    print(f"Output files are located in: {TARGET_DIR}")

if __name__ == "__main__":
    main()
