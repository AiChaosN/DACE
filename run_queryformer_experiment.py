import lightning.pytorch as pl
from torch.utils.data import DataLoader
import argparse
import os
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from lightning.pytorch.callbacks import ModelCheckpoint

# Import from DACE modules
from model import PL_DACE, DACELora, PLTrainer
from utils import ROOT_DIR, set_seed, plan_parameters, load_json
from plan_utils import (
    process_plans, 
    prepare_dataset, 
    add_numerical_scalers, 
    get_op_name_to_one_hot,
    get_plan_encoding
)

def load_custom_plans(configs, op_name_to_one_hot, feature_statistics):
    """
    Loads train.json, job_light.json, synthetic.json
    Returns a dictionary of plans meta info.
    """
    data_dir = os.path.join(ROOT_DIR, "data/queryformer_dace")
    
    files = {
        "train": "train.json",
        "job_light": "job_light.json",
        "synthetic": "synthetic.json"
    }
    
    # Map dataset names to IDs
    db_ids = {
        "train": 0,
        "job_light": 1,
        "synthetic": 2
    }
    
    all_plans_meta = {
        "train": [],
        "job_light": [],
        "synthetic": []
    }
    
    for name, filename in files.items():
        path = os.path.join(data_dir, filename)
        print(f"Loading and processing {path}...")
        with open(path, 'r') as f:
            plans = json.load(f)
            
        for plan in plans:
            plan["database_id"] = db_ids[name]
            plan_meta = get_plan_encoding(
                plan, configs, op_name_to_one_hot, plan_parameters, feature_statistics
            )
            all_plans_meta[name].append(plan_meta)
            
    return all_plans_meta

def train_and_test(configs):
    # 1. Load Statistics
    print("Loading statistics...")
    statistics_path = "data/queryformer_dace/statistics.json"
    feature_statistics = load_json(os.path.join(ROOT_DIR, statistics_path))
    add_numerical_scalers(feature_statistics)
    op_name_to_one_hot = get_op_name_to_one_hot(feature_statistics)
    
    # 2. Load Plans
    print("Loading plans...")
    plans_data = load_custom_plans(configs, op_name_to_one_hot, feature_statistics)
    
    # Strip database_id (last element) as prepare_dataset expects 4 elements
    train_full = [item[:-1] for item in plans_data["train"]]
    job_light = [item[:-1] for item in plans_data["job_light"]]
    synthetic = [item[:-1] for item in plans_data["synthetic"]]
    
    # 3. Split Train Data (70% Train, 30% Test)
    train_subset, test_subset_30 = train_test_split(
        train_full, test_size=0.3, random_state=configs["random_seed"]
    )
    
    print(f"Data Split:")
    print(f"  Training Set (from train.json): {len(train_subset)}")
    print(f"  Test Set 1 (from train.json):   {len(test_subset_30)}")
    print(f"  Test Set 2 (job_light.json):    {len(job_light)}")
    print(f"  Test Set 3 (synthetic.json):    {len(synthetic)}")
    
    # 4. Prepare Datasets for Training
    # Split train_subset into train and val (90/10)
    train_data, val_data = train_test_split(
        train_subset, test_size=0.1, random_state=configs["random_seed"]
    )
    
    train_dataset = prepare_dataset(train_data)
    val_dataset = prepare_dataset(val_data)
    
    batch_size = configs["batch_size"]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 5. Setup Model
    model = DACELora(
        configs["node_length"],
        configs["hidden_dim"],
        1,
        configs["mlp_activation"],
        configs["transformer_activation"],
        configs["mlp_dropout"],
        configs["transformer_dropout"],
    )
    
    pl_model = PL_DACE(model)
    
    csv_logger = pl.loggers.CSVLogger(save_dir=os.path.join(ROOT_DIR, "Results"), name="dace_queryformer_logs")
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(ROOT_DIR, "Results", "checkpoints_queryformer"),
        filename="DACE-QueryFormer",
        save_top_k=1,
        mode="min"
    )
    
    trainer = PLTrainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[0] if torch.cuda.is_available() else 1,
        enable_progress_bar=configs["progress_bar"],
        max_epochs=configs["max_epoch"],
        logger=csv_logger,
        callbacks=[checkpoint_callback],
    )
    
    # 6. Train
    print("Starting training...")
    trainer.fit(pl_model, train_dataloader, val_dataloader)
    
    # Load best model for testing
    print(f"Loading best checkpoint from {checkpoint_callback.best_model_path}")
    best_model = PL_DACE.load_from_checkpoint(checkpoint_callback.best_model_path, model=model)
    
    # Move model to device manually as PLTrainer.test overrides standard behavior and doesn't handle placement automatically
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model.to(device)
    
    # 7. Evaluation
    test_sets = [
        ("Test Set 1 (train 30%)", test_subset_30),
        ("Test Set 2 (job_light)", job_light),
        ("Test Set 3 (synthetic)", synthetic)
    ]
    
    results = {}
    
    for name, data in test_sets:
        if len(data) == 0:
            print(f"Skipping {name} (empty)")
            continue
            
        print(f"\nEvaluating on {name}...")
        dataset = prepare_dataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # Use PLTrainer's test method which calculates Q-Error
        # Note: custom PLTrainer.test ignores ckpt_path, it uses the passed model
        test_metrics = trainer.test(best_model, dataloaders=dataloader)
        results[name] = test_metrics

    # Save results
    with open(os.path.join(ROOT_DIR, "Results", "queryformer_results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    print("\nResults saved to Results/queryformer_results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--node_length", type=int, default=22)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--pad_length", type=int, default=50)
    parser.add_argument("--max_epoch", type=int, default=10) # 10 Epochs
    parser.add_argument("--loss_weight", type=float, default=0.5)
    parser.add_argument("--mlp_activation", type=str, default="ReLU")
    parser.add_argument("--transformer_activation", type=str, default="gelu")
    parser.add_argument("--mlp_dropout", type=float, default=0.3)
    parser.add_argument("--transformer_dropout", type=float, default=0.2)
    parser.add_argument("--progress_bar", action="store_true", default=True)
    
    args = parser.parse_args()
    configs = vars(args)
    configs["max_runtime"] = 30000
    
    set_seed(configs["random_seed"])
    
    train_and_test(configs)
