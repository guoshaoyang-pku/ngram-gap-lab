# 假说 H-DILUTE：碰撞稀释的逐 context 记忆 × backbone 读出放大（v0，2026-08-30）

> 状态：**假设**（待挑战）。本笔记是主报告 §6 的完整版，供后续 challenge 迭代。
> 证据输入全部来自已登记 run 与 `data/freq_index.npz`（seed-42 1x shard，sha256 见脚本输出）；
> 复现脚本：`docs/plot_scripts/plot_v5_theory_alpha_collapse.py`、`plot_v5_interference_model.py`。
> 产物：`docs/figs/theory/fig_v5_interference_model_vs_data.png`、`fig_v5_zipf_context_alpha.png`、
> `docs/figs/main/fig_v5_pass_collapse.png`、`theory_zipf_triangle.csv`、`theory_interference_scan.csv`、
> `docs/appendices/s1_scaling_three_axis/s1_dose_points_128x.csv`。

## 1. 模型

对每个 branch（bigram / trigram）：

```
Gap(R, passes, t) ≈ A(t, passes) · Σ_c m_c · κ(f_c) · s_c(R)
s_c(R) = f_c / (f_c + T/R)
```

- `m_c = f_c / T`：context c 的 token 质量份额（T 为该 branch 训练总 token 数）。
- `κ(f)`：逐 context 记忆核。经验输入取频率轴实测 `f^{-β}`（bigram β=0.253、trigram β=0.318，
  `s1v5_128_frequency_main` step-1000，token-mass-weighted 局部斜率）。**微观起源未推导**。
- `s_c(R)`：hash 碰撞下的写入份额。推导（非拟合）：RMSProp 无动量把每次命中的更新步长近似
  归一化到常数，因此一行收敛到共享该行的各 context 残差的**流量加权平均**；c 在自己行内的
  流量份额为 `f_c/(f_c + T_other)`，均场取 `T_other ≈ T/R`。被稀释掉的部分对 train/val 近似
  同错（同一行输出对两侧一样），对 gap 贡献 ≈ 0。
- `A(t, passes)`：backbone 读出放大。online 语义下 pass 1 恒等于 0（首次见样本时行内容
  尚未写入 / 无收益）；随重复训练近线性增长并逐渐凹弯；受 backbone wd=0.1 约束应最终饱和。

## 2. 已闭合的数值检验

1. **f 轴 → R 轴闭合**（图 `fig_v5_interference_model_vs_data.png`）：把 β 与真实计数直方图
   `n(f)` 代入卷积 `G(R) ∝ Σ_f n(f)·(f/T)·f^{-β}·f/(f+T/R)`，在与实测相同的拟合窗口内得
   局部斜率 **bigram 0.567 / trigram 0.648**，实测 net-gap 斜率 0.576 / 0.665；trigram 曲线
   在 R=2e2–2e6 全程贴合（每 branch 只有一个自由幅度参数），大 R 饱和弯头位置一致。
   → 频率指数与表大小指数确有确定关系，但它是「β + 真实 n(f) + 碰撞份额」的卷积结果。
2. **朴素代数版被证伪**（图 `fig_v5_zipf_context_alpha.png`、`theory_zipf_triangle.csv`）：
   理想 Zipf `q_r ∝ r^{-α}` 下可解析出 `γ = 1 − α(1−β)`。实测 rank–frequency 局部
   α_bigram=0.994（窗口 2e3–2e5，R²=.994）、α_trigram=0.841（1e5–9.3e5，R²=.997），代入
   得 γ=0.26/0.43 ≠ 实测 0.58/0.66；闭合所需 α 为 0.57/0.49。真实计数分布有曲率
   （bigram 窗口敏感性 0.67–1.28），故只有数值卷积版成立。双对数「线性」只在中段成立
   即来源于此。
3. **pass 数是主状态变量**（图 `fig_v5_pass_collapse.png`）：dose 128× 批、>1×L4 wrap 点、
   10-epoch 长 replay 在「完成 pass 数」横轴上折叠；dose 6× 在 2000 步 = 0.99 pass 处 gap
   穿零（online 语义直接预言）；0.75× 的 7.9-pass 点 7.21 vs replay 8-pass 7.14。
   0.25×（23.7 passes，10.90）低于线性外推 → 高 pass 端凹弯继续发展。
4. **epoch-length 轴勘误**：>1×L4 的点共用 shard-1 频率索引，是 wrap-around 重放
   （2×L4 3-epoch 终点 5.582 ≈ replay 6-pass 5.609），属 pass 数轴；≤1×L4 才是真
   epoch-length 变量（gap 随 L 从 3.55 降到 2.47）。旧“U 形 + 顶点 0.41×L4”作废。
5. **dose 批次勘误**：`nglab*_input_v5_freq10` 是 2× 批（config `table_lr_scale=2.0`，
   slope −1.727）；128× 权威批为 `nglab*_input_v5_128x_freq10`（slope −1.176，R²=.899，
   n=10，≤5×），登记于 `s1_dose_points_128x.csv`。

## 3. 挑战清单（按优先级）

1. **κ(f) 微观起源**：采样方差论给 1/f、等步长 CLT 论给 f^{-1/2}，均比实测 β≈1/4、1/3 陡。
   候选：小 f 端被条件熵 H_c 截断（few-shot 记忆收益有上限）、大 f 端趋向 (S_c−1)/f 的
   交叉区，在 H_c、S_c 异质的 context 混合下呈现中间表观斜率。
   检验（零 GPU）：按 next-token 条件熵分层重画 g(f)；看超大 f 端局部斜率是否变陡；
   看 g(f) 在不同 pass 数快照下是否近似沿 `e·f` 平移（已有 step-337/674/1000 快照）。
2. **静态份额 ≠ 动态 mask**：卷积预测 trigram f≤8 份额约 25%（f≤200 约 84%），而 mask_low
   f≤8 动态去掉约 74%（2.945→0.765，2× 批）。差异指向 mask 后的再平衡 / backbone scar。
3. **A(t,passes) 无动力学**：freeze_table（3.452 > control 2.724）说明后期增长不需要表写入；
   freeze_backbone（1.230）说明需要 backbone 更新；mask_high t=1 残留 1.927 说明存在
   不对称恢复 / scar 项。需要显式 backbone-表共适应动力学。
4. **share¹ vs share²**：share¹ 是 RMSProp 等步长的自然结果，但二次损失下的 share² 未排除
   （`theory_interference_scan.csv` 中 hard-threshold 变体亦记录在案）。
5. **β 的 regime 依赖**：2× 时代旧描述拟合更陡（~0.56–0.75）；需同一诊断在 2× 与 128×
   已有 freq JSONL 上重测（零 GPU）。

## 4. 可证伪预测

- (a) 把 3 个 pass 的数据全局混匀（保持每 context 总命中数、打破 epoch 分块）后，终点 gap
  与分块 replay 相同，epoch 齿消失。若混匀显著改变终点 gap → 模型缺少「重复间隔」变量。
- (b) 频率轴局部 β 应随 R 系统变化（小 R 更平：稀释把低频端压掉）。
- (c) 增大 backbone wd 压低高 pass 端 gap；wd=0 时高 pass 端更接近线性。
- (d) 5-gram / MLP backbone 的 (β, γ) 也应由各自计数直方图卷积闭合（α_n 随 n 变平）。

## 5. 与用户直觉的对应

- 「支撑维度 / 每参数样本数」→ 本模型的 `κ(f)` 与 s_c 份额：表侧每行少量样本、可结实下降；
  backbone 参数叠加（superposition）表现为 A 增长缓慢、且 no-gram 对照 gap 很小。
- 「提高 table LR 不能一步到位、gap 收敛到与 pass 数有关的值」→ online 语义 + A(t,passes)：
  表内容在 1 个 pass 内就近饱和（LR≥16× 后 LR 不再是瓶颈），后续增长主要是 backbone 侧
  放大与再见样本的记忆收益，只能随 pass 积累。
- 「熵正则化力 / 重复次数决定平衡位置」→ 无需显式正则项：等步长写入 + 碰撞流量稀释 + wd
  约束的读出增益，共同给出随重复次数移动的准平衡位置（尚未严格化，见挑战 3）。

---

## v2 修订（2026-08-30 晚，吸收第二 reviewer 的四点批评）

**证据等级下调**：v0 的「已闭合」降级为「同数据强一致性检验」。四点批评全部接受并已处置：
(a) κ 来自 fixed 4-batch train probe 而非主口径 online（登记为口径差异；probe/online 粗桶斜率
bigram −0.214 / trigram −0.436 vs 探针几何桶 −0.253/−0.318）；(b) κ 在 R=2^20 双表条件下测量、
被用于单表 R 轴 → 有重复计入风险；(c) β 不是常数：单表 runs 实测 β(R) 随 R 从 ~0.26 升到 ~0.46
（`beta_by_table_R.csv`，另一 agent 零 GPU 结果）——方向恰是 v0 预测 (b)，但定量形式否定均场；
(d) 频率幂律口径不唯一。

**新的零 GPU 判决实验（本仓库数据，脚本 `plot_v5_missing_mass_kernel.py`、`plot_v5_dilution_surface.py`）**：

1. **κ(f) 微观起源候选（H-KAPPA）**：κ_true(f) ≈ B·M(f) + V·min(S_eff,f)/f。
   - M(f) = Good–Turing 缺失延续质量 = N1(f)/(f·n(f))，直接由语料组合结构给出（bigram 支路：
     延续类型=trigram 类型，f=1 时 M=1.000、f=8:0.510、f=128:0.269）。这是「支撑维度」直觉的
     严格化：M(f) = 条件分布支撑随样本数增长（Heaps）的边际率，理想幂律尾 a 下 M∝f^{−(1−1/a)}。
   - 第二项 min(S_eff,f)/f 是插件估计方差项（(S−1)/f 型），S_eff(f)=该 f 层平均延续种类数。
   - 拟合（bigram，19 个几何桶，R=2^20 双表 run，两个幅度参数）：B=4.17、V=3.40，linR²=0.928；
     纯 M(f) 单幅度 linR²=0.796。窗口 [4,4096] 斜率：数据 −0.441、两分量核同窗口一致（见图）。
   - **被否定的稀释函数形式**：均场 s=f/(f+T/R)（f=1 处预测压制 47×，实测无此塌陷）与
     分布式 Poisson 行内竞争 s_dist（f=1 处预测 0.16×，同样否定）。
2. **经验稀释面 s_emp(f,R)**（62 个单表 bigram run 相对 R_ref=2.35e6 归一）：
   - 压制在 f 上近乎平坦、主要由负载 K/R 决定（Q3 收拢）；load 10→1000 的 log-log 斜率 ≈ −0.59，
     与 R 轴实测 0.576 吻合 → **table-size 幂律主要是 f-平坦负载压制 a(K/R) 的形状**，
     不是 v0 设想的「核质量逐步纳入」。f-平坦性指向 backbone 侧通道增益随 SNR 校准
     （与 freeze/scar 证据同源），行内流量竞争不是主机制。
   - s_emp 有轻微 f 倾斜（中 R 低 f 略强压制），定性解释 β(R) 漂移；大 R 端实测 β≈0.44–0.46
     趋近两分量核的窗口斜率 −0.441 ✓（核形状在大 R 极限恢复）。
3. **v2 形式**：Gap(R,f,t) ≈ A(t,passes) · a_emp(K/R) · [B·M(f)+V·S_eff(f)/f] · m_f，
   其中 a_emp 由稀释面 CSV 插值；开放项变为 a(load) 的动力学起源与 A 的时间结构。

**待跑判决**（已登记）：~~trigram 支路 M(f)~~ ✅ 已完成（4-gram 计数集群 CPU job，
`theory_missing_mass_trigram.csv`，commit 2ecf5e0：M(f) 在 [1,100] 斜率 −0.319 ≈ 实测 β −0.318）；
causal_dynamics 批（freeze_backbone e1/e2/e3、freeze_table e2、wd 0/0.3、control，2022 步 6 pass）
标定 A 动力学——running（§38）；backbone-LR 判决批 running（§39）；混匀（pass 交错）实验需
train.py 新旗标，规格已登记、下一批实现。

---

## H-KAPPA 完整推导（2026-08-31 深夜补，用户要求）

### 0. 论证形态：两个独立测量相等，不是 curve fit

- **训练侧**：per-context gap 随 f 的衰减指数 β，由 run `s1v5_128_frequency_main`（seed 42，
  input 双表 R=2^20，step-1000 exact_freq_loss，token-mass 加权局部斜率）量得：
  bigram −0.253、trigram −0.318。
- **语料侧**：缺失延续质量 M(f) 的斜率，由 `data/freq_index.npz`（seed-42 shard-1）纯计数得到：
  bigram −0.241（窗口 [1,100]）/ −0.258（[4,4096]）；trigram −0.319（[1,100]）。
- H-KAPPA 断言两者相等；实测命中。**M(f) 零训练、零 GPU、零自由形状参数**；
  全部拟合自由度 = 幅度（两分量核 B、V 两个数，或单幅度 C）。若 M(f) 形状是拟合的，
  这只是一个 fit；因为它是数出来的，这是解释。

### 1. 定义（bigram 支路；trigram 把延续类型换成 4-gram 类型）

- context c：前 n−1 个 token；train 中总出现次数 f_c。
- 延续类型：t = (c, next-token)，即下一级 n-gram 类型（bigram 支路 = trigram 类型，
  前缀索引 `tri_keys // 8192` 精确可查）。
- N1(c)：c 的延续类型中在 train 恰出现 1 次的个数（singleton 类型数）。
- 频率层 L_f = {c : f_c = f}，n(f) = |L_f|，N1(f) = Σ_{c∈L_f} N1(c)。
- **M(f) = N1(f) / (f · n(f))** = 层内平均的 N1(c)/f。

### 2. Good–Turing：M(f) ≈ P(val 下文是 c 没见过的新类型)

设真条件分布 p(·|c) 的类型概率 {q_i}。c 在 train 被抽 f 次（跨 pass 近似 i.i.d.）：

- val 下文类型 train 未见过：P_novel(f) = Σ_i q_i (1−q_i)^f。
- singleton 计数期望：E[N1(c)] = Σ_i f·q_i (1−q_i)^{f−1}。
- 故 E[N1(c)/f] = Σ_i q_i (1−q_i)^{f−1} ≈ Σ_i q_i (1−q_i)^f = P_novel(f)：
  逐项相对差 O(q_i)，重尾下由 q_i ≪ 1 的尾部类型主导，近似成立（标准 GT 缺失质量估计）。
- **f=1 sanity**：c 只出现 1 次 ⇒ 其唯一延续类型必为 singleton ⇒ M(1) = 1。
  数据：bigram M(1)=1.000、trigram M(1)=1.000 ✓。
- 实测：bigram M(8)=0.510、M(128)=0.269；trigram M(8)=0.535、M(128)=0.296。
  f 越大，延续支撑按 Heaps 律次线性增长，新鲜下文供给按幂律（非指数）枯竭。

### 3. 幂律尾 ⇒ β = 1 − 1/a（β 的微观来源）

设单 context 延续分布尾部 q_r ∝ r^{−a}（a>1，Z 为归一常数）：

```
P_novel(f) = Σ_r q_r (1−q_r)^f ≈ (1/Z) ∫ r^{−a} exp(−f·r^{−a}/Z) dr
换元 u = f·r^{−a}/Z  (r = (f/(Zu))^{1/a})：
P_novel(f) ≈ (Γ(1−1/a)/a) · Z^{−1/a} · f^{−(1−1/a)}
```

即 **M(f) ∝ f^{−(1−1/a)}，β = 1 − 1/a**，与优化器、表大小、训练时间均无关。
反解 a = 1/(1−β)：

- bigram：β=0.253 ⇒ a ≈ 1.34 ≈ 4/3
- trigram：β=0.318 ⇒ a ≈ 1.47 ≈ 3/2

解读：context 越长，条件分布越尖（尾部越陡、a 越大），延续支撑长得越慢，
新鲜延续消耗越快 ⇒ β 越大。**β 是语言条件分布尖度的投影**——「支撑维度」直觉的严格化。

### 4. 两分量核：κ(f) = B·M(f) + V·min(S_eff,f)/f

- **第一项（未见延续，主导）**：val 下文类型 train 完全没见过 ⇒ 行对该类型分配 ≈ 0 质量 ⇒ 惩罚大。
- **第二项（已见延续的插件方差）**：行内容是 f 个样本的经验分布；即便 val 下文类型已被见过，
  行的质量分配仍有 O((S−1)/f) 量级误差。S_eff(f) = 层内平均延续种类数（index 直接可数）；
  min(·,f) 防止 f < S_eff 时越界。
- 拟合（bigram，19 个几何桶，val-token 加权线性最小二乘）：**B=4.17、V=3.40，linR²=0.928**；
  纯 M 单幅度版 linR²=0.796。窗口 [4,4096] 实测数据斜率 −0.441，两分量核同窗口一致
  （图 `fig_v5_missing_mass_kernel.png` 左面板：蓝点=实测，绿线=C·M(f)，
  红虚线=被否定的碰撞稀释版，紫线=两分量核）。

### 5. 与总公式的接口、边界与复现

- 在 Gap(R,f,t) ≈ A(t,passes) · a_emp(K/R) · κ(f) · m_f 中，κ 的形状全部由 §1–4 的语料量给出；
  A、a_emp 的幅度另行标定（稀释面 CSV / causal 批）。
- 边界（诚实清单）：
  1. M(f) 斜率依赖窗口（bigram −0.22~−0.26；trigram −0.19~−0.32），[1,100] token 质量主区
     命中最干净；**β 不作普适常数使用**（另见 β(R) 漂移，§v2-2）。
  2. 第一项只解释 gap 大头：方差项（V）、backbone scar（mask_high t=1 残留 1.927）、
     读出放大 A 均在核外。
  3. 目前为「同数据强一致性检验」；升级为因果判据依赖混匀实验与 causal_dynamics 批（§38）。
  4. E[N1/f] ≈ P_novel 的近似在 f 大、头部质量重时偏差增大（§2 换元的紧性条件）。
- 复现：`docs/plot_scripts/plot_v5_missing_mass_kernel.py`（bigram）+
  `docs/plot_scripts/compute_fourgram_missing_mass.py`（trigram，集群 CPU job）
  → `docs/figs/theory/theory_missing_mass_{bigram,trigram}.csv`。
