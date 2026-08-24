# Cluster patch guide (retired)

> 本页只保留迁移后的历史说明。不要在集群上手工 patch `train.py`，也不要
> 根据旧行号或旧工作区坐标操作。当前唯一支持的入口是
> `ngram5_freq_gap/cluster/run_on_cluster.sh`。

## 当前运行契约

`run_on_cluster.sh` 会把本仓库的 `code/train.py` 同步到目标仓库副本，
再同步 `ngram5_freq_gap/`，最后启动本地的 `trainer.py`。`model.py` 只从
这份同步的主线模型加载 `NanoGPT`，因此不需要第二份 backbone，也不需要
修改集群上的模型源码。

运行前设置目标机的仓库、缓存和 SSH 别名：

```bash
export NGLAB_SSH_HOST=cluster-alias
export NGLAB_CLUSTER_ROOT=/path/to/ngram-gap-lab
export NGLAB_CLUSTER_CACHE=/path/to/autoresearch-cache
bash ngram5_freq_gap/cluster/run_on_cluster.sh 0.0 0 10
```

`NGLAB_CLUSTER_ROOT` 必须是目标机上本仓库副本的根目录；
`NGLAB_CLUSTER_CACHE` 必须包含兼容的 tokenizer/parquet cache。结果默认写入
目标仓库的 `data/runs_fixed/`，每个 run 使用带 `_fixed` 后缀的目录。
需要复用已审核数据集时，在目标机上将
`NGRAM5_DATA_DIR_OVERRIDE` 设置为该数据集目录，并确认其中包含
`meta.json`、`metadata.json` 和 `exact_ngram_counts.npz`。

## 数据与频率分解

`lib.py` 已注册 `ngram5_blocks` 数据模式，`trainer.py` 使用
`FivegramIndex` 对 exact train-epoch context count 做分解。五元组频率
使用 bucket id 作为索引，不能用可能溢出 64 位整数的
`vocab_size**5` base encoding。数据生成器和训练器共享同一哈希定义；
因此不应再把 hash bucket occupancy 当作理论频率。

## 验证

首先运行短 smoke：

```bash
bash ngram5_freq_gap/cluster/run_on_cluster.sh 0.0 0 10
```

成功的结果目录应包含 `run_contract.json`、`summary.json`、
`allgram_frequency_decomposition.jsonl` 以及（启用 full trace 时）
`trace_manifest.jsonl` 和 `batch_trace/`。检查 contract 中的模型 source、
`table_betas`、数据频率定义和结果路径后，才可以启动长实验。
