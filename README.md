# DACE

## 运行 run_workload1_forGNTO.py 所需环境与步骤

若只需运行 `run_workload1_forGNTO.py` 进行 DACE 在 Workload1 上的训练与测试，需满足以下条件：

### 1. 环境要求

- **Python**: 3.9+（建议 3.9.18）
- **依赖包**：在项目根目录执行
  ```bash
  pip install -r requirements.txt
  ```
  主要依赖：`torch`, `lightning`/`pytorch_lightning`, `scikit-learn`, `numpy`, `tqdm`, `loralib` 等

### 2. 配置 ROOT_DIR

在 `utils.py` 中将 `ROOT_DIR` 修改为本地 DACE 项目根目录的绝对路径，例如：
```python
ROOT_DIR = "/path/to/your/DACE"
```

### 3. 数据准备

**数据来源**：Workload1 数据集来自 [OSF parsed_plans](https://osf.io/rb5tn/overview/runs/parsed_plans)

**目录结构**：需在 `data/workload1/` 下放置 20 个数据集的 JSON 文件，例如：
- `accidents.json`, `airline.json`, `baseball.json`, `basketball.json`, `carcinogenesis.json`, `consumer.json`, `credit.json`, `employee.json`, `fhnk.json`, `financial.json`
- `geneea.json`, `genome.json`, `hepatitis.json`, `imdb_full.json`, `movielens.json`, `seznam.json`, `ssb.json`, `tournament.json`, `tpc_h.json`, `walmart.json`

**预处理**（若仅有原始 `.json`，需先执行）：
```bash
python setup.py --filter_plans --get_statistic
```
会生成 `*_filted.json` 和 `statistics.json`。

### 4. 运行命令

在 DACE 项目根目录下执行：
```bash
python run_workload1_forGNTO.py
```

可选参数示例：
```bash
python run_workload1_forGNTO.py --max_epoch 10 --batch_size 512 --random_seed 123
```

### 5. 输出说明

- 训练日志：`Results/dace_workload1_logs/`
- 模型检查点：`Results/checkpoints_workload1/`
- 测试结果：`Results/0227_dace_workload1_results.json`

---

## Compare GNTO
### 数据集使用:
Workload1数据集来源:https://osf.io/rb5tn/overview/runs/parsed_plans
Workload1数据集 (10个数据集: accidnet, airline, baseball, basketball, carcinogenesis, consumer, credit, employee, fhnk, financial)

### 运行命令:
To compare DACE and GNTO, run:
```bash
python run_workload1_forGNTO.py
```
This will train DACE on 10 databases and test on 10 databases.
The results will be saved in the results folder.
The results will be compared with GNTO.
The results will be saved in the results folder.

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