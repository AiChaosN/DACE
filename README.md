# DACE
# Comparison: DACE vs. GNTO

## Environment Setup and Steps for Running `run_workload1_forGNTO.py`

To train and test DACE on Workload1 using `run_workload1_forGNTO.py`, please ensure the following requirements are met:

### 1. Environment Requirements

- **Python**: 3.9+ (Recommended: 3.9.18)
- **Dependencies**: From the project root directory, install:
  ```bash
  pip install -r requirements.txt
  ```
  Key dependencies include: `torch`, `lightning` / `pytorch_lightning`, `scikit-learn`, `numpy`, `tqdm`, `loralib`, etc.

### 2. Configure `ROOT_DIR`

In `utils.py`, set `ROOT_DIR` to the absolute path of your local DACE project directory. For example:
```python
ROOT_DIR = "/path/to/your/DACE"
```

### 3. Data Preparation

**Data Source**: The Workload1 datasets are available from [OSF parsed_plans](https://osf.io/rb5tn/overview/runs/parsed_plans).

**Directory Structure**: Place the 20 JSON datasets under `data/workload1/`, e.g.:
- `accidents.json`, `airline.json`, `baseball.json`, `basketball.json`, `carcinogenesis.json`, `consumer.json`, `credit.json`, `employee.json`, `fhnk.json`, `financial.json`
- `geneea.json`, `genome.json`, `hepatitis.json`, `imdb_full.json`, `movielens.json`, `seznam.json`, `ssb.json`, `tournament.json`, `tpc_h.json`, `walmart.json`

**Preprocessing** (If you only have raw `.json` files, run the following first):
```bash
python setup.py --filter_plans --get_statistic
```
This will generate `*_filted.json` and `statistics.json`.

### 4. Run the Script

From the DACE project root directory, run:
```bash
python run_workload1_forGNTO.py
```

Example for running with optional arguments:
```bash
python run_workload1_forGNTO.py --max_epoch 10 --batch_size 512 --random_seed 123
```

### 5. Output Description

- Training logs: `Results/dace_workload1_logs/`
- Model checkpoints: `Results/checkpoints_workload1/`
- Test results: `Results/0227_dace_workload1_results.json`

---

## Overview
DACE: A Database-Agnostic Cost Estimator.

## Getting Started

### Prerequisites
- Python 3.9.18
- Required Python packages (see `requirements.txt`)

### Installation
Clone the repository and install the dependencies:
```bash
git clone git@github.com:liang-zibo/DACE.git
cd dace
pip install -r requirements.txt
```

### Download DACE Data
Before running the code, please download the data from this
[data repository](https://figshare.com/s/58a0e03829db15bef555) and put them in the data folder.

## Usage
### Modify ROOT_DIR
Modify ROOT_DIR in utils.py to your own path.

### Filtering Plans and Gathering Statistics
To filter out plans and gather statistical data, run:
```bash
python setup.py --filter_plans --get_statistic
```

### Get plan encodings
To get plan encodings, run:
```bash
python run.py --process_plans
```

### Testing All Databases
To sequentially use each database as a test set while treating the remaining databases as a training set, execute:
```bash
python run.py --test_all
```

### Testing on IMDB Dataset
To test and evaluate DACE's performance on the IMDB dataset (dataset ID: 13), without including any knowledge from the IMDB dataset in the training set, use:
```bash
python run.py --test_database_ids 13
cd data
mv DACE.ckpt DACE_imdb.ckpt
```

### Direct Testing on Workloads
To directly test DACE as a pre-trained estimator on job-light, scale, and synthetic workloads:
```bash
python run_tuning.py
```

### Tuning and Testing on Workloads
For fine-tuning and testing DACE as a pre-trained estimator on job-light, scale, and synthetic workloads:
```bash
python run_tuning.py --tune
```

## Contact

If you have any questions about the code, please email [zibo_liang@outlook.com](mailto:zibo_liang@outlook.com)