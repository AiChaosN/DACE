import torch
import os
from torch.utils.data import DataLoader
from model import DACELora, PL_DACE, PLTrainer
from utils import ROOT_DIR, load_json, plan_parameters, set_seed
from plan_utils import process_plans, prepare_dataset, add_numerical_scalers, get_op_name_to_one_hot

def evaluate(checkpoint_path, test_db_id):
    # Load configs (mimicking run.py defaults)
    configs = {
        "random_seed": 123,
        "node_length": 22,
        "hidden_dim": 128,
        "mlp_activation": "ReLU",
        "transformer_activation": "gelu",
        "mlp_dropout": 0.3,
        "transformer_dropout": 0.2,
        "batch_size": 512,
        "pad_length": 50,
        "max_runtime": 30000,
        "loss_weight": 0.5,
        "statistics_path": "data/workload1/statistics.json",
        "plans_dir": "data/workload1",
        "test_database_ids": [test_db_id]
    }
    
    set_seed(configs["random_seed"])
    
    # 1. Prepare Data
    statistics_file_path = configs["statistics_path"]
    feature_statistics = load_json(os.path.join(ROOT_DIR, statistics_file_path))
    add_numerical_scalers(feature_statistics)
    op_name_to_one_hot = get_op_name_to_one_hot(feature_statistics)
    
    # This will load from pkl if it exists
    plans_meta = process_plans(configs, op_name_to_one_hot, plan_parameters, feature_statistics)
    
    test_data = []
    # plans_meta item: (seq_encoding, run_times, attention_mask, loss_mask, database_id)
    for plan_meta in plans_meta:
        if plan_meta[-1] == test_db_id:
            test_data.append(plan_meta[:-1])
            
    print(f"Found {len(test_data)} plans for database ID {test_db_id}")
    if len(test_data) == 0:
        print("No test data found. Check database ID.")
        return

    test_dataset = prepare_dataset(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=configs["batch_size"], shuffle=False)
    
    # 2. Load Model
    base_model = DACELora(
        configs["node_length"],
        configs["hidden_dim"],
        1,
        configs["mlp_activation"],
        configs["transformer_activation"],
        configs["mlp_dropout"],
        configs["transformer_dropout"],
    )
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading checkpoint from {checkpoint_path}")
    # PL_DACE.load_from_checkpoint handles loading weights into the wrapped model
    pl_model = PL_DACE.load_from_checkpoint(checkpoint_path, model=base_model)
    
    # 3. Test
    # Using cpu or gpu based on availability, similar to run.py but simplified
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = PLTrainer(accelerator=accelerator, devices=1, logger=False) # Disable logger to print to stdout
    
    print(f"Starting testing on device: {accelerator}")
    trainer.test(pl_model, dataloaders=test_dataloader)

if __name__ == "__main__":
    # IMDB database ID is 13
    evaluate("checkpoints/DACE-v2.ckpt", 13)
