import json
import os

ROOT_DIR = "/home/AiChaosN/Project/Workspace/01_Research/DACE"
OLD_STATS_PATH = os.path.join(ROOT_DIR, "data/workload1/statistics.json")
NEW_STATS_PATH = os.path.join(ROOT_DIR, "data/queryformer_dace/statistics.json")

def update_statistics():
    with open(OLD_STATS_PATH, 'r') as f:
        stats = json.load(f)
    
    # Add BitmapAnd
    current_max_idx = max(stats["node_types"]["value_dict"].values())
    if "BitmapAnd" not in stats["node_types"]["value_dict"]:
        stats["node_types"]["value_dict"]["BitmapAnd"] = current_max_idx + 1
        print(f"Added BitmapAnd with index {current_max_idx + 1}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(NEW_STATS_PATH), exist_ok=True)
    
    with open(NEW_STATS_PATH, 'w') as f:
        json.dump(stats, f)
    print(f"Saved updated statistics to {NEW_STATS_PATH}")

if __name__ == "__main__":
    update_statistics()
