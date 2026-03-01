# 数据集使用:
1. Queryformer数据集 (三个数据集: train, job_light, synthetic)

2. Workload1数据集 (10个数据集: accidnet, airline, baseball, basketball, carcinogenesis, consumer, credit, employee, fhnk, financial)

# 运行命令:
1. workload1数据集:
```sh
# 删除已有的plans_meta.pkl文件
rm data/workload1/plans_meta.pkl

# 过滤掉运行时间过短的查询计划
python setup.py --filter_plans

# 基于过滤后的数据生成统计信息（均值、分布、算子类型等）
python setup.py --get_statistic

# 运行workload1数据集训练
python run_workload1.py
```

2. Queryformer数据集:
```sh
# 转换Queryformer数据集为DACE数据集
python convert_queryformer_to_dace.py

# 运行Queryformer数据集训练
python run_queryformer_experiment.py
```

# 数据集结果目录:
Results/queryformer_results.json
Results/workload1_results.json
