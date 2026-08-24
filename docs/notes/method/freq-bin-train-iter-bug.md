# Handover：`freq_bin_loss` 诊断复用 `train_iter` 的 bug

> 状态：**bug 已定位、修复已应用（未 commit）、冒烟验证未完成、全部实验待重跑**
> 本文件供 reviewer 执行检查与重跑使用。所有旧实验数据均受此 bug 影响，修复后需全部重跑。

## 0. TL;DR

- Bug：`code/train.py` 的 train 侧频率分桶诊断 `evaluate_freq_bins(model, train_iter, ...)` 把**真正的训练迭代器**传给诊断函数，每 `freq_eval_interval` 步额外消费 5 个训练 batch（`n_batches=4` 时因 off-by-one 实耗 5 个）。
- 后果 1（数据流被污染）：这些 batch 被跳过、永不参与优化，模型实际训练轨迹 ≠ 名义 step 数。
- 后果 2（标签虚高）：诊断同时推进 `train_ds._epoch/_batch_in_epoch`，epoch 计数虚高；若启用 `--lr_schedule_epochs` 还会扭曲 LR 进度。
- 影响范围：所有启用 `--freq_index` 的运行（**全部** run script 均启用）→ 即全部 `nglab*` 历史实验。
- 修复：已应用（本地 + 360-2 远程），核心是「独立诊断迭代器 + 修正 off-by-one」。**未 commit、未完整验证、未重跑**。

## 1. Bug 细节

### 1.1 症状

- epoch 边界提前：名义约 `337 step/epoch`（1x，bs72），实际提前约 `0.3-3` 个 epoch（取决于 `freq_eval_interval`）。
- `freq_bin_loss.jsonl` / `train_log.jsonl` 的 `epoch` 字段虚高（两者都取 `train_ds._epoch + 1`，一起被推高）。

### 1.2 根因（修复前逻辑）

```python
# train.py:845  训练与诊断共用同一个迭代器
train_iter = train_ds.iter_batches(device)

# 训练循环（grad_accum 内）每步消费 1 个训练 batch
inp, tgt = next(train_iter)

# 每 freq_eval_interval 步，诊断直接消费同一个 train_iter
train_freq = evaluate_freq_bins(model, train_iter, freq_index_obj,
                                args.freq_eval_batches, cfg.vocab_size)

# evaluate_freq_bins 内部：先取 batch 再判断 → off-by-one
for i, (inp, tgt) in enumerate(loader):
    if i >= n_batches:   # n_batches=4 时第 5 个 batch 已取出但被丢弃
        break
```

- `TokenizedShardDataset.iter_batches`（`train.py:673-687`）在完整回放一遍训练分片后 `self._epoch += 1`，每个 batch `self._batch_in_epoch += 1`。诊断消费 batch 时同样推进这些计数，因此训练侧的 epoch/LR 进度被诊断"免费加速"。
- val 侧不受影响：`fixed_freq_val_batches` 在启动时缓存（`train.py:855`），诊断每次复用同一批固定 val 数据。

### 1.3 量化影响（以 `nglab1x_v10_input` 为例）

- 配置：2000 步，`--freq_eval_interval 10 --freq_eval_batches 4`，1x = shards 1,2 = 24264 chunks → 337 batch/epoch。
- 诊断消耗：200 次 eval × 5 = 1000 batch ≈ **2.97 epoch**。
- 模型真实训练：2000 batch ≈ **5.93 epoch**；但日志末尾记录 `epoch: 9`（(2000+1000)/337 ≈ 8.9）。
- 通用公式（修复前）：`记录epoch = floor((step + eval次数×5) / batches_per_epoch) + 1`。

## 2. 证据

- 历史 `data/runs/nglab1x_v10_input/train_log.jsonl`：最后一条 `step 2000, epoch 9`；`freq_bin_loss.jsonl` 共 200 行。
- 历史 `data/runs/nglab1x_e6`（1400 步）：末尾 `epoch 7` vs 真实 ≈4.15 epoch。
- 历史 `data/runs/nglab5x_input_fv`、`nglab6x_input_fv`、`nglab8x_input_fv` 等同样受影响。
- 所有历史 run 的 `train.log` 都有 `[nglab] freq-bin eval enabled`，即诊断路径全部开启。

## 3. 已应用的修复（reviewer 需复核）

### 3.1 修复内容

1. **off-by-one 修复**（`evaluate_freq_bins`，`train.py:746-758`）：改用 `loader_iter = iter(loader)` + `for _ in range(n_batches)` + `StopIteration` 处理，恰好消费 `n_batches` 个 batch。对 val 侧的 list 输入同样适用。
2. **独立 train 诊断迭代器**（`train.py:891-896`）：在启用 `--freq_index` 时新建独立的 `TokenizedShardDataset` 实例 `freq_train_ds` 并取其 `freq_train_iter`。独立实例**不共享** `train_ds._epoch/_batch_in_epoch`，诊断既不消费训练流、也不推进训练计数。
3. **调用点**（`train.py:947`）：`train_iter` → `freq_train_iter`。

查看改动：`git diff code/train.py`（相对 commit `363b0ee`）。

### 3.2 注意：工作树有其他未提交改动（非本 bug 修复）

`git status` 显示 `code/train.py` 工作树未提交，其中包含**早于本次修复**的本地改动，reviewer 不要回退：

- `--lr_schedule_epochs` epoch 锚定 LR 特性（`Config.lr_schedule_epochs`、`epoch_progress()`、`_batch_in_epoch` 计数）；
- `MixedOptimizer` 的 `b2 = self.table_betas[1]` 修复。

当前所有 run script 均未传 `--lr_schedule_epochs`（默认 0，按 step 进度），因此重跑不受该特性影响。

### 3.3 同步状态

| 位置 | 路径 | 状态 |
|---|---|---|
| 本地（git 仓库） | `code/train.py` | 已修复，未 commit |
| 远程副本（**非 git**，普通拷贝） | 由 `NGLAB_ROOT/code/train.py` 指定 | 已同步，含修复 |

远程无版本控制，重跑前必须重新校验一致性（见 §5）。

## 4. 修复后的预期行为

- epoch 标签回归真实训练进度：1x 2000 步 → `floor(2000/337)+1 = 6`（修复前为 9）。
- 训练流不再被跳过：每个优化 step 恰好消费 1 个训练 batch。
- train 侧 freq-bin 评测的是**最近已训练过的批次**：独立迭代器每 interval 消费 4 个 batch < 训练 10 个，永远滞后于训练流 → 符合「train loss 是历史量」的语义。
- `freq_bin_loss.jsonl` 与 `train_log.jsonl` 的 `epoch` 字段仍一致（都取 `train_ds._epoch+1`），只是现在真实。
- val 侧、`train_log` 的瞬时 train_loss 语义均不变。

## 5. Reviewer 检查清单 —— 代码复核

- [ ] 本地与远程 `code/train.py` 内容一致（`diff` / `md5`）。
- [ ] `evaluate_freq_bins` 只消费恰好 `n_batches` 个 batch，无 off-by-one。
- [ ] 全文件无残留 `evaluate_freq_bins(model, train_iter, ...)`（应只剩 `freq_train_iter`）。
- [ ] `freq_train_ds` 是独立的 `TokenizedShardDataset` 实例（不共享 `train_ds` 状态）。
- [ ] val 侧仍使用 `fixed_freq_val_batches`，未被改动。
- [ ] 工作树中 `lr_schedule_epochs` / `table_betas` 改动为历史遗留，确认未误伤。

## 6. Reviewer 检查清单 —— 冒烟验证（360-2）

> **修正（2026-08-11）**：原 handover 此处期望值有误（用了 ~42 batch/epoch，实际 shard 62 = 6066 chunks / 72 = **84 batch/epoch**）。以下为修正后的配置和期望值。

判别设计：用 0.25x shard（62），84 batch/epoch，400 步可跨过多次 epoch，新旧代码 epoch 差异明显。

```bash
ssh cluster-alias
cd /path/to/ngram-gap-lab
mkdir -p data/runs_fixed/smoke_fixed_verify
CUDA_VISIBLE_DEVICES=7 python3 -u code/train.py \
  --run_id smoke_fixed_verify --injection_position input --steps 400 --seed 42 \
  --data_dir data/tokenized --out_dir data/runs_fixed \
  --train_shards 62 --val_shards 2,3,4,5,6,7,8,9,10,6542 \
  --device_batch_size 72 --total_batch_size 147456 \
  --val_interval 10 --val_batches 4 --table_norm_interval 10 --lr 0.004 \
  --enable_unigram 0 --enable_bigram 1 --enable_trigram 1 \
  --n_layer 8 --n_head 6 --n_embd 768 --vocab_size 8192 --sequence_len 2048 \
  --freq_eval_interval 10 --freq_eval_batches 4 \
  --freq_index data/freq_index_train0_25x.npz
```

验证点：

- [ ] 运行不崩溃，`freq_bin_loss.jsonl` 恰好 40 行（400/10）。
- [ ] `train_log.jsonl` 末尾 `epoch` 应为 **5**（400/84 ≈ 4.76 → `_epoch=4` → 记录 5）；修复前会是 **6**（(400+200)/84 ≈ 7.14 → 记录 8，但因 off-by-one 实际消费 5×40=200 → (400+200)/84=7.14 → epoch=8）。
- [ ] `freq_bin_loss.jsonl` 每行 `epoch` 与 `train_log.jsonl` 对应行一致。
- [ ] （强验证）`nglab1x_v10_input_nofb_fixed`（不传 `--freq_index`）与 `nglab1x_v10_input_fixed` 的 `train_loss` 序列完全一致（排除诊断干扰）。

## 7. 重跑计划（全部历史实验）

### 7.1 原则

- 必须使用修复后的 `train.py`。
- 新 run_id 一律加 `_fixed` 后缀（如 `nglab1x_v10_input_fixed`），**不得覆盖旧 run 目录**；旧数据保留作对比。
- 每个 run 的 CLI 必须与旧 run 完全一致，以 launch 脚本为准；无脚本的 run 从 `summary.json` + `train.log` 重建配置并记录到 `docs/RERUN_MANIFEST.md`。
- 按 family 分批，每批占满 360-2 的 8 张 H200（当前空闲），跑完立即对比归档。

### 7.2 实验清单（family → 脚本 → 代表 run_id）

| family | 脚本 | 代表 run_id |
|---|---|---|
| 注入位置 v10（1x，2000 步） | `code/cluster/run_injpos.sh` | `nglab1x_v10_v` / `_y` / `_input` / `_nogram` |
| epoch 尺度 v10（2x / 0.5x，2000 步） | `code/cluster/run_epoch_scale_v10.sh` | `nglab2x_input_v10_fv` / `nglab0_5x_input_fv` |
| shard sweep 360（5x/6x/8x，2000 步） | `code/cluster/run_shard_sweep_360.sh` | `nglab5x_input_fv` / `_6x` / `_8x` |
| 其他 shard sweep（input_fv 系列） | `code/cluster/run_shard_sweep.sh` / `_v2.sh` | `nglab0_25x_input_fv` … `nglab4x_input_fv_v3` |
| e6 系列（0.25x…3x） | 历史脚本未归档，需从 run 重建 | `nglab0_25x_e6` … `nglab3x_e6` |
| table-opt（1x/2x，adamw/rmsprop/sgd、b2 变体） | `code/cluster/run_table_opt.sh` / `_2x.sh` | `nglab1x_opt_*` / `nglab2x_opt_*` |
| 短 epoch b2=0.99（0.25x/0.5x，2000 步） | `code/cluster/run_epoch_short_b2.sh` | `nglab025x_b2_099` / `nglab05x_b2_099` |
| 早期 2x v/y/input | `code/cluster/run_train2x.sh` | `nglab2x_v` / `_y` / `_input` |

完整 run 清单：历史 `data/runs/`（58 目录）与远程结果（23 目录）的并集。重跑结果统一回收到 `data/runs_fixed/`。

### 7.3 对比方法

- 每对 (old, `_fixed`) 对比：`train_log.jsonl`（`epoch` 序列、`train_loss`、`val_loss`、`gap`、`lr_mult`）与 `freq_bin_loss.jsonl`（分桶 train/val loss）。
- 预期差异：epoch 标签整体变小（约 -3 @2000 步）；训练数据流变化后 loss/gap 曲线可能整体平移或形状微变——这是本次重跑要观察的核心。
- 产出：每个 family 一份 step 对齐的对比摘要，记录到 `docs/RERUN_MANIFEST.md`。

## 8. 环境与注意事项

- 集群：通过 SSH 别名连接，具体仓库路径由 `NGLAB_ROOT` 指定。
- 数据与索引：`data/tokenized/shard_*.bin` + `data/freq_index*.npz` 在远程齐全；shard 60/61/62/63/64 为 0.5x/0.25x 分片。
- 算力：2000 步 1x ≈ 75 min/run（H200）；并行 8 个，全量重跑墙钟约 10-15 h。
- 旧 run 数据（含 bug 版本）**一律保留**，重跑确认后再决定归档。
- 修复尚未 `git commit`；验证通过后建议先 commit（本地仓库），再同步远程。

## 9. 主 agent 重跑记录（2026-08-11 20:40 起）

### 9.1 发现的额外问题

1. **e6 系列 val_shards 与 train 重叠（已修复）**：原 handover 重跑计划中 e6 系列统一用 `STD_VAL = "2,3,4,5,6,7,8,9,10,6542"`，但 e6 的 2x/2.5x/3x train shards 包含 2 和 3，导致 val 重叠。已修正为按 train shards 动态选择 val（与原始 e6 run 一致）。
2. **`table_betas` b2 语义变化**：`_table_rmsprop_step` 里 `b2 = self.ngram_beta2` → `b2 = self.table_betas[1]`。默认 rmsprop（不传 `--table_betas`）b2 仍为 0.999，行为不变；但传了非默认 `--table_betas` 的 rmsprop arm（如 `rmsprop_2x_b2_099`），旧 run 实际跑 b2=0.999（`--table_betas` 被忽略），新 run 才真正应用指定 b2。这些 arm 的 `_fixed` 版本才是真正的实验数据。
3. **handover agent 的 `nglab1x_e6_fixed` 参数错误**：handover agent 给该 run 加了 `--lr_schedule_epochs 6`（非标准参数），与用户要求"无多余参数"冲突。该 run 跑完后已被主 agent 用正确参数（无 `--lr_schedule_epochs`，steps=1685=5 real epoch）重跑覆盖。

### 9.2 重跑配置

- **e6 系列**：改为 5 real epoch（`steps = bpe * 5`），不再用旧的 buggy-7-epoch steps。
- **所有 run**：统一 `freq_eval_interval=10, val_interval=10`，无 `--lr_schedule_epochs`，无其他多余参数。
- **run_id**：一律加 `_fixed` 后缀，不覆盖旧数据。
- **早期 run**（nglab_input/v/y/nogram, nglab2x_v/y/input）已被 v10 系列取代，不重跑。

### 9.3 GPU 分配

- 360-2 GPU 0-3：handover agent 的 4 个先行实验（5x/6x/8x + nglab1x_e6_fixed）
- 360-2 GPU 4-7：主 agent 的 22 个 run（4 队列，每队列 5-6 个 run）
- 360-1 GPU 1,3,6,7：主 agent 的 25 个 run（4 队列，每队列 5-7 个 run）
- 总计：47 个 run 在 8 GPU 上并行，预计墙钟约 2.8 小时

### 9.4 验证 run

- `smoke_fixed_verify`（shard 62, 400 步）：在 360-2 GPU 4 队列末尾，验证 fixed epoch=5
- `nglab1x_v10_input_nofb_fixed`（不传 `--freq_index`）：在 360-1 GPU 3 队列，验证诊断不影响训练流

### 9.5 相关文件

- `code/cluster/rerun_all.py`：实验清单生成器
- `code/cluster/assign_gpus.py`：GPU 队列分配器
- `code/cluster/launch_360_1.sh` / `launch_360_2.sh`：自动生成的 launch 脚本
