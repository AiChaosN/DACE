import lightning.pytorch as pl
from torch.utils.data import DataLoader
import argparse
import os
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from lightning.pytorch.callbacks import ModelCheckpoint
from tqdm import tqdm

# Import from DACE modules
from model import PL_DACE, DACELora, PLTrainer
from utils import ROOT_DIR, set_seed, plan_parameters, load_json
from plan_utils import (
    prepare_dataset, 
    add_numerical_scalers, 
    get_op_name_to_one_hot,
    get_plan_encoding,
    q_error_np,
    print_qerrors
)

def load_plans_adapted(path, configs, op_name_to_one_hot, feature_statistics, database_id=0):
    data = load_json(path)
    plans_meta = []
    
    # Handle the structure of queryformer_dace JSONs: [{"Plan": ...}, ...]
    # Or nested lists depending on previous steps.
    # We verified job_light.json has [{"Plan":...}] structure via read_file (text),
    # but let's be robust.
    
    desc = f"Processing {os.path.basename(path)}"
    for item in tqdm(data, desc=desc):
        # Unwrap if necessary
        plan_content = item
        while isinstance(plan_content, list):
            plan_content = plan_content[0]
        
        # Now plan_content should be {"Plan": ...}
        if "Plan" not in plan_content:
            continue
            
        plan_content["database_id"] = database_id
        
        # Skip run_time filter to use all data
        
        plan_meta = get_plan_encoding(
            plan_content, configs, op_name_to_one_hot, plan_parameters, feature_statistics
        )
        # (seq_encoding, run_times, attention_mask, loss_mask, database_id)
        plans_meta.append(plan_meta[:-1])
        
    return plans_meta

def main(configs):
    # 1. Prepare Statistics and Configs
    print("Loading statistics...")
    statistics_path = os.path.join(ROOT_DIR, "data/queryformer_dace/statistics.json")
    feature_statistics = load_json(statistics_path)
    add_numerical_scalers(feature_statistics)
    op_name_to_one_hot = get_op_name_to_one_hot(feature_statistics)
    
    # Calculate correct node_length
    # Ops + Cost + Rows
    actual_node_length = len(feature_statistics["node_types"]["value_dict"]) + 2
    configs["node_length"] = actual_node_length
    print(f"Detected node_length: {actual_node_length}")
    
    # 2. Prepare Data
    train_path = os.path.join(ROOT_DIR, "data/queryformer_dace/train.json")
    print(f"Loading training data from {train_path}")
    train_plans_meta = load_plans_adapted(train_path, configs, op_name_to_one_hot, feature_statistics)
    
    # Split Train/Val
    train_data, val_data = train_test_split(
        train_plans_meta, test_size=0.1, random_state=configs["random_seed"]
    )
    
    train_dataset = prepare_dataset(train_data)
    val_dataset = prepare_dataset(val_data)
    
    batch_size = configs["batch_size"]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 3. Model Setup
    model = DACELora(
        configs["node_length"], 
        configs["hidden_dim"], 
        1, 
        configs["mlp_activation"],
        configs["transformer_activation"],
        configs["mlp_dropout"],
        configs["transformer_dropout"]
    )
    
    pl_model = PL_DACE(model)
    
    # Results Directory
    results_dir = os.path.join(ROOT_DIR, "Results")
    os.makedirs(results_dir, exist_ok=True)
    
    csv_logger = pl.loggers.CSVLogger(save_dir=results_dir, name="dace_tuning_logs")
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(results_dir, "checkpoints_tuning"),
        filename="DACE_Correct",
        save_top_k=1,
        mode="min"
    )
    
    trainer = PLTrainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[0] if torch.cuda.is_available() else 1,
        max_epochs=configs["max_epoch"],
        logger=csv_logger,
        callbacks=[checkpoint_callback],
    )
    
    # 4. Training
    print("Starting training...")
    trainer.fit(pl_model, train_dataloader, val_dataloader)
    
    print(f"Best checkpoint saved at: {checkpoint_callback.best_model_path}")
    
    # 5. Testing
    print("Loading best model for testing...")
    best_model = PL_DACE.load_from_checkpoint(checkpoint_callback.best_model_path, model=model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model.to(device)
    best_model.eval()
    
    test_workloads = ["job_light", "synthetic"]
    results = {}
    
    for workload in test_workloads:
        print(f"Testing on {workload}...")
        path = os.path.join(ROOT_DIR, f"data/queryformer_dace/{workload}.json")
        plans_meta = load_plans_adapted(path, configs, op_name_to_one_hot, feature_statistics)
        
        if not plans_meta:
            print(f"No plans for {workload}")
            continue
            
        test_dataset = prepare_dataset(plans_meta)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        preds = []
        targets = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                seqs, masks, loss_masks, times = [b.to(device) for b in batch]
                output = best_model(seqs, masks)
                
                preds.append(output[:, 0].cpu().numpy())
                targets.append(times[:, 0].cpu().numpy())
        
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        
        # Calculate Q-Error
        # Values are normalized 0-1 (approx). Ratio q-error works fine.
        # But we should ensure no exact zeros.
        preds = np.maximum(preds, 1e-7)
        targets = np.maximum(targets, 1e-7)
        
        q_errors = q_error_np(preds, targets)
        print_qerrors(q_errors)
        
        results[workload] = {
            "mean": float(np.mean(q_errors)),
            "50th": float(np.quantile(q_errors, 0.5)),
            "90th": float(np.quantile(q_errors, 0.9)),
            "95th": float(np.quantile(q_errors, 0.95)),
            "99th": float(np.quantile(q_errors, 0.99)),
            "max": float(np.max(q_errors))
        }
    
    # Save Results
    res_path = os.path.join(results_dir, "tuning_results_correct.json")
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {res_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--pad_length", type=int, default=50)
    parser.add_argument("--max_epoch", type=int, default=10)
    parser.add_argument("--loss_weight", type=float, default=0.5)
    parser.add_argument("--mlp_activation", type=str, default="ReLU")
    parser.add_argument("--transformer_activation", type=str, default="gelu")
    parser.add_argument("--mlp_dropout", type=float, default=0.3)
    parser.add_argument("--transformer_dropout", type=float, default=0.2)
    
    args = parser.parse_args()
    configs = vars(args)
    configs["max_runtime"] = 30000
    
    set_seed(configs["random_seed"])
    
    main(configs)
