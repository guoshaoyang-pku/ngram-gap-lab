# n-gram Gap Plotting Guide

本目录是 `ngram-gap-lab` 的唯一 canonical 作图代码区。所有图都从
`data/runs/<run_id>/` 的 JSONL 统计读取，不在 HTML 中手写实验数值；
源码留在 `docs/plot_scripts/`，生成的 HTML/SVG 写入 `docs/figs/`，
不写回 `code/` 或本目录。

## 数据源

| 文件 | 内容 | 主要用途 |
|---|---|---|
| `train_log.jsonl` | step、train loss、validation loss、global gap、epoch | 训练曲线与 epoch replay 边界 |
| `table_norm.jsonl` | 每个 n-gram table 的 RMS/norm | table memory growth 与 gap 对齐 |
| `freq_bin_loss.jsonl` | 每个 context branch、split、frequency bucket 的 token count、fraction、mean loss、total contribution | frequency-bin loss/gap 分解 |

其中：

```text
global gap = validation loss - train loss
bucket mean loss = bucket 内 token loss 的平均值
bucket total contribution = bucket fraction × bucket mean loss
```

`freq_bin_loss.jsonl` 的 loss 是基于 unreduced per-token cross-entropy
聚合出来的：先保留每个 token 的 loss，再按真实 context 的 hit-count bucket
累积。当前运行产物保存的是每个 bucket 的 token count、fraction、mean loss
和 total contribution，不保存数千万个 token 的逐 token loss 原数组。

## 脚本索引

### 主博客图表

`gen_all_figures.py` 是当前 canonical generator：

| 函数 | 输出 | 作图思想 |
|---|---|---|
| `gen_fig_gap_loss` | `fig_gap_loss.html` | v/y/input 的 global train/val loss 与 gap；用 Plotly 图例控制曲线 |
| `gen_fig_loss_norm` | `fig_loss_norm.html` | 在一张 Plotly 图中用多个 y 轴同时显示 loss、gap 与 bigram/trigram table RMS |
| `gen_fig_gap_by_freq` | `fig_gap_by_freq.html` | 每个 frequency bucket 的时间序列；可切换 per-token loss、gap、total contribution，以及 bigram/trigram |
| `gen_fig_gap_log` | `fig_gap_vs_frequency_loglog.html`、`fig_gap_vs_frequency_logx.html` | 以 bucket hit-count 几何中点作 x；log-log 版本观察定量关系，log-x/linear-y 版本直接读取 gap 大小 |
| `gen_static_frequency_figures` | `fig_freq_bigram.svg`、`fig_freq_trigram.svg` | 静态 SVG 单图双 y 轴：左轴 train/val token fraction 双柱，右轴 train/val mean loss 与 final gap 曲线（train 无 token 的 novel 桶断线）；不画 total contribution 柱图 |

`gen_fig_hitcount_dist` 仍可用于 standalone 分布诊断，但不属于当前
主博客的 frequency → gap 三图。正确画法是：train/val token fraction
用 bar，最终 checkpoint 的 per-bin gap 用 secondary-axis line；不能把
fraction 或 cumulative fraction 柱图称为 gap 图。

log 图只使用 train 和 validation 都有 token 且 gap 为正的 bucket。
`novel` 表示 train hit count 为 0；由于 train 侧没有对应 token loss，
不能定义标准的 `val loss - train loss`，因此只保留在 raw/fraction 图中，
不进入 gap 曲线。

### 其他作图脚本

| 脚本 | 用途 |
|---|---|
| `gen_injpos_plot.py` | 早期注入点 Plotly 图，保留作 provenance |
| `build_injpos_data_json.py` | 为博客首页 summary cards 和 norm table 生成轻量 JSON |
| `gen_epoch_scale_figs.py` | 0.5x/1x/2x epoch length 的 train/val/gap 对比 |
| `gen_all_figures_v10.py` | v10/fixed-val 实验的独立版本，使用环境变量指定图目录 |

## 生成流程

在 repo 根目录执行：

```bash
python3 docs/plot_scripts/gen_all_figures.py
```

默认输出到 `docs/figs/`。同步生成到 sibling GitHub Pages repo 时：

```bash
NGRAM_GAP_BLOG_FIGS_DIR=/path/to/guoshaoyang-pku.github.io/blogs/ngram-gap-mechanism-guide \
python3 docs/plot_scripts/gen_all_figures.py
```

生成后检查：

1. frequency 图是否显示全部 bucket，且 categorical axis 没有把 `6-10` 误读成数值；
2. Plotly 图例点击是否能隐藏/恢复曲线；
3. log 图是否排除了 novel、undefined gap 和非正 gap；
4. HTML、SVG 引用是否存在；
5. `fig_loss_norm.html` 是否只有一个 Plotly figure，并同时包含 loss、gap、table RMS；
6. `git diff --check` 和浏览器 console 是否无错误。

### Blog embedding: no scrolling

博客中的所有图必须完整展开，不允许 iframe 内部滚动或裁切：

```html
<iframe class="chart-frame loss-norm"
        src="fig_loss_norm.html"
        scrolling="no"
        title="Loss、gap 与 table RMS 的时间对齐"></iframe>
```

为每一种图设置明确的 CSS 高度，并以浏览器实际测量值为准：

```javascript
Array.from(document.querySelectorAll("iframe")).map((frame) => ({
  src: frame.src,
  frameHeight: frame.clientHeight,
  contentHeight: frame.contentDocument?.documentElement?.scrollHeight ?? null
}));
```

只有在 `contentHeight <= frameHeight` 时才算通过。移动端 media query
也必须保留足够高度；不能用统一的 `480px` 覆盖多面板图，也不能为了
消除滚动条而删除 Plotly 图例、控制按钮或数据面板。

当前 canonical 主线使用 validation 与 frequency evaluation 每 10 步（v10）。
早期 `nglab_{v,y,input}` 记录使用每 50 步，仅作为已完成的历史基线，不应
与 v10 曲线混称。

## 可复现性约束

- 不要从旧宽 bucket 插值或伪拆出更细的 exact hit-count loss。
- 需要 exact hit-count 曲线时，重新构建 exact context frequency index，
  用 unreduced token loss 做在线或离线聚合。
- 不要把 generated HTML/SVG 写回 `code/`；源码留在本目录，生成物留在
  `docs/figs/`，原始统计留在 gitignored `data/`。
- 主博客当前只嵌入 `fig_gap_loss.html`、`fig_loss_norm.html`、
  `fig_gap_by_freq.html`、`fig_gap_vs_frequency_logx.html` 和
  `fig_gap_vs_frequency_loglog.html`；`fig_hitcount_dist.html` 仅作可选
  standalone diagnostic，不应重新加入主 frequency → gap 章节。