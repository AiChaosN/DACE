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
import random
from model import PL_DACE, DACELora, PLTrainer
from utils import ROOT_DIR, set_seed, plan_parameters, load_json, workloads, getModelSize
from plan_utils import (
    process_plans, 
    prepare_dataset, 
    add_numerical_scalers, 
    get_op_name_to_one_hot,
    q_error_np,
    print_qerrors
)

def main(configs):
    # 1. Configs
    print("Loading statistics from workload1...")
    statistics_path = os.path.join(ROOT_DIR, "data/workload1/statistics.json")
    feature_statistics = load_json(statistics_path)
    add_numerical_scalers(feature_statistics)
    op_name_to_one_hot = get_op_name_to_one_hot(feature_statistics)
    
    # Calculate node_length
    configs["node_length"] = len(feature_statistics["node_types"]["value_dict"]) + 2
    print(f"Node length: {configs['node_length']}")

    # 2. Data Preparation
    # This will load/generate plans_meta.pkl for workload1
    # plans_meta item: (seq_encoding, run_times, attention_mask, loss_mask, database_id)
    print("Processing plans (this may take a while if not cached)...")
    plans_meta = process_plans(
        configs, 
        op_name_to_one_hot, 
        plan_parameters, 
        feature_statistics,
        pre_process_path="data/workload1/plans_meta.pkl"
    )
    
    # Random select 10 databases for training and 10 for testing
    train_db_ids = [3, 10, 11, 12, 13, 14, 15, 16, 18, 19]
    test_db_ids = [0, 1, 2, 4, 5, 6, 7, 8, 9, 17]

    print(f"Training Databases: {[workloads[i] for i in train_db_ids]}")
    print(f"Testing Databases: {[workloads[i] for i in test_db_ids]}")
    
    train_data_full = []
    test_data_by_db = {i: [] for i in test_db_ids}
    
    for item in plans_meta:
        db_id = item[-1]
        data_point = item[:-1] # Remove db_id for dataset
        
        if db_id in train_db_ids:
            train_data_full.append(data_point)
        elif db_id in test_db_ids:
            test_data_by_db[db_id].append(data_point)
            
    print(f"Total Train samples: {len(train_data_full)}")
    for db_id in test_db_ids:
        print(f"Test samples (DB {db_id} - {workloads[db_id]}): {len(test_data_by_db[db_id])}")
        
    # Split Train into Train/Val
    train_data, val_data = train_test_split(
        train_data_full, test_size=0.1, random_state=configs["random_seed"]
    )
    
    train_dataset = prepare_dataset(train_data)
    val_dataset = prepare_dataset(val_data)
    
    batch_size = configs["batch_size"]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 3. Model
    model = DACELora(
        configs["node_length"], 
        configs["hidden_dim"], 
        1, 
        configs["mlp_activation"],
        configs["transformer_activation"],
        configs["mlp_dropout"],
        configs["transformer_dropout"]
    )

    #---------- Print model size ----------
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    getModelSize(model)

    #   | Name  | Type     | Params
    # -----------------------------------
    # 0 | model | DACELora | 23.4 K
    # -----------------------------------
    # 12.3 K    Trainable params
    # 11.1 K    Non-trainable params
    # 23.4 K    Total params
    # 0.093     Total estimated model params size (MB)

    pl_model = PL_DACE(model)
    
    results_dir = os.path.join(ROOT_DIR, "Results")
    os.makedirs(results_dir, exist_ok=True)
    
    csv_logger = pl.loggers.CSVLogger(save_dir=results_dir, name="dace_workload1_logs")
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(results_dir, "checkpoints_workload1"),
        filename="DACE_Workload1",
        save_top_k=1,
        mode="min"
    )
    
    trainer = PLTrainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[0] if torch.cuda.is_available() else 1,
        max_epochs=configs["max_epoch"],
        logger=csv_logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True
    )
    
    # 4. Train
    print("Starting training...")
    trainer.fit(pl_model, train_dataloader, val_dataloader)
    
    # 5. Test
    print(f"Loading best model from {checkpoint_callback.best_model_path}")
    best_model = PL_DACE.load_from_checkpoint(checkpoint_callback.best_model_path, model=model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model.to(device)
    best_model.eval()
    
    results = {}
    
    for db_id in test_db_ids:
        db_name = workloads[db_id]
        print(f"Testing on {db_name}...")
        data = test_data_by_db[db_id]
        
        if not data:
            print(f"No data for {db_name}")
            continue
            
        dataset = prepare_dataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        preds = []
        targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                seqs, masks, loss_masks, times = [b.to(device) for b in batch]
                output = best_model(seqs, masks)
                preds.append(output[:, 0].cpu().numpy())
                targets.append(times[:, 0].cpu().numpy())
                
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        
        preds = np.maximum(preds, 1e-7)
        targets = np.maximum(targets, 1e-7)
        
        q_errors = q_error_np(preds, targets)
        
        results[db_name] = {
            "mean": float(np.mean(q_errors)),
            "50th": float(np.quantile(q_errors, 0.5)),
            "90th": float(np.quantile(q_errors, 0.9)),
            "95th": float(np.quantile(q_errors, 0.95)),
            "99th": float(np.quantile(q_errors, 0.99)),
            "max": float(np.max(q_errors))
        }
        print(f"Results for {db_name}: Median={results[db_name]['50th']:.4f}, Mean={results[db_name]['mean']:.4f}")

    # Save
    json_path = os.path.join(results_dir, "0227_dace_workload1_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default args from run.py
    random_seed = random.randint(1, 1000)
    parser.add_argument("--random_seed", type=int, default=random_seed) # 123
    parser.add_argument("--node_length", type=int, default=22) # 更具统计信息写上node type的种类  
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--pad_length", type=int, default=50)
    parser.add_argument("--max_epoch", type=int, default=10) # 训练次数 
    parser.add_argument("--loss_weight", type=float, default=0.5)
    parser.add_argument("--mlp_activation", type=str, default="ReLU")
    parser.add_argument("--transformer_activation", type=str, default="gelu")
    parser.add_argument("--mlp_dropout", type=float, default=0.3)
    parser.add_argument("--transformer_dropout", type=float, default=0.2)
    parser.add_argument("--plans_dir", type=str, default="data/workload1")
    
    args = parser.parse_args()
    configs = vars(args)
    configs["max_runtime"] = 30000
    
    set_seed(configs["random_seed"])
    main(configs)
