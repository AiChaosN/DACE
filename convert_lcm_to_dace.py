# 主要用于将lcm-eval的数据集转换为DACE的数据集.

import os
import json
import glob
from tqdm import tqdm

# 配置路径
ROOT_DIR = "/home/AiChaosN/Project/Workspace"
LCM_EVAL_DIR = os.path.join(ROOT_DIR, "02_Research/lcm-eval/data/runs/parsed_plans")
DACE_DATA_DIR = os.path.join(ROOT_DIR, "01_Research/DACE/data/workload1")

# DACE 需要的数据库列表 (来自 utils.py)
WORKLOADS = [
    "accidents", "airline", "baseball", "basketball", "carcinogenesis",
    "consumer", "credit", "employee", "fhnk", "financial",
    "geneea", "genome", "hepatitis", "imdb_full", "movielens",
    "seznam", "ssb", "tournament", "tpc_h", "walmart"
]

# 映射 lcm-eval 的键名到 DACE (Postgres原始) 键名
def convert_node(lcm_node):
    params = lcm_node.get("plan_parameters", {})
    
    dace_node = {
        "Node Type": params.get("op_name", "Unknown"),
        "Total Cost": params.get("est_cost", 0.0),
        "Plan Rows": params.get("est_card", 0.0),
        "Actual Total Time": params.get("act_time", 0.0),
        # DACE 有时也需要 Startup Cost
        "Startup Cost": params.get("est_startup_cost", 0.0)
    }
    
    # 递归处理子节点
    children = lcm_node.get("children", [])
    if children:
        dace_node["Plans"] = [convert_node(child) for child in children]
        
    return dace_node

def convert_dataset(dataset_name):
    # 寻找源文件，优先使用 100k workload
    src_dir = os.path.join(LCM_EVAL_DIR, dataset_name)
    if not os.path.exists(src_dir):
        # 尝试一些命名差异
        if dataset_name == "imdb_full" and os.path.exists(os.path.join(LCM_EVAL_DIR, "imdb")):
             src_dir = os.path.join(LCM_EVAL_DIR, "imdb")
        else:
            print(f"Warning: Source directory not found for {dataset_name}")
            return

    # 查找主要的 json 文件
    json_files = glob.glob(os.path.join(src_dir, "*workload*.json"))
    # 简单的启发式：找包含 100k 的，如果没有就找任意 json
    target_file = None
    for f in json_files:
        if "100k" in f:
            target_file = f
            break
    if not target_file and json_files:
        target_file = json_files[0]
        
    if not target_file:
        print(f"Warning: No plan json found for {dataset_name}")
        return

    print(f"Converting {dataset_name} from {target_file}...")
    
    try:
        with open(target_file, 'r') as f:
            # lcm-eval 的文件通常是一个巨大的单行 JSON 对象
            data = json.load(f)
            
        plans_list = data.get("parsed_plans", [])
        
        dace_formatted_plans = []
        for plan in plans_list:
            # 转换树结构
            dace_plan_tree = convert_node(plan)
            
            # DACE setup.py 期望的结构:
            # item[0][0][0]["Plan"]
            # 所以我们需要: [[[ {"Plan": ...} ]]]
            wrapped_plan = [[[ {"Plan": dace_plan_tree} ]]]
            dace_formatted_plans.append(wrapped_plan)
        
        output_path = os.path.join(DACE_DATA_DIR, f"{dataset_name}.json")
        
        # 写入整个列表为一个 JSON 对象
        with open(output_path, 'w') as out_f:
            json.dump(dace_formatted_plans, out_f)
                
    except Exception as e:
        print(f"Error converting {dataset_name}: {e}")

def main():
    if not os.path.exists(DACE_DATA_DIR):
        os.makedirs(DACE_DATA_DIR)
        
    for workload in tqdm(WORKLOADS):
        convert_dataset(workload)

if __name__ == "__main__":
    main()
