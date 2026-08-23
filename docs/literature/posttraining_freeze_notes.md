# 后训练冻结记忆组件：证据面（subagent Leibniz，2026-08-01）

**证据面一：重复数据 / 多 epoch 导致过拟合与记忆化**
- Xue et al.（NeurIPS 2023）：token 受限下多 epoch 重复引发过拟合与 multi-epoch degradation，模型越大越敏感。
- Tirumala et al.（NeurIPS 2022）：「记忆化先于过拟合」——模型先精确记忆重复样本，容量越大记忆越多、遗忘越慢；解释显式记忆组件在 replay 下被低频样例填满、gap 放大的机制。
- Gao et al.（ICML 2023）：RLHF 过度优化 = 代理奖励随 KL 的扩展律；PPO 多 epoch/多次采样时代理奖励上升、真实奖励先升后降、KL 上漂直至崩溃；与 SFT 多 epoch 重复数据同源。

**证据面二：冻结部分组件是既有且有效的实践**
- kNN-LM（Khandelwal et al., ICLR 2020）：冻结预训练 LM、记忆完全非参数化、线性插值接入 → 「记忆只读、网络可训」先例。
- BitFit（Ben Zaken et al., ACL 2022）：冻结绝大部分参数、仅训 bias 可逼近全量微调。
- LoRA Learns Less and Forgets Less（Biderman et al., TMLR 2024）：受限更新在指令微调中「学得更少但忘得更少」。

**证据面三：记忆/检索组件的泛化短板**
- Nishida et al.（NAACL Findings 2025）：kNN-LM 增益几乎只来自高频 token，对低频/长尾无改善甚至更差——与我们观察到的低频 context 过拟合一致。
- Geng et al.（2024）Great Memory, Shallow Reasoning：kNN 记忆擅长密集任务但在推理任务上退化。

**结论支撑**：可训练 n-gram 记忆在 SFT/RLHF 多 epoch 重复数据下应当冻结或转只读检索，避免把后训练分布偏差固化进记忆。

引用已复核（2026-08-02），见 `references.bib`：Xue et al. NeurIPS 2023 (arXiv:2305.13230)；Gao et al. ICML 2023 (arXiv:2210.10760)；Khandelwal et al. ICLR 2020 (arXiv:1911.00172)；Nishida et al. NAACL Findings 2025 (arXiv:2503.22426)；Tirumala et al. NeurIPS 2022 (arXiv:2205.10770)；另补 BitFit ACL 2022 (arXiv:2106.10199)、LoRA Learns Less TMLR 2024 (arXiv:2405.09673)、Great Memory NAACL 2025 (arXiv:2408.11815)。已并入 plan-2 §2.4.3。

**可引用条目**：
1. To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis — Fuzhao Xue et al., NeurIPS 2023
2. Scaling Laws for Reward Model Overoptimization — Leo Gao, John Schulman, Jacob Hilton, ICML 2023
3. Generalization through Memorization: Nearest Neighbor Language Models — Khandelwal et al., ICLR 2020
4. Long-Tail Crisis in Nearest Neighbor Language Models — Nishida et al., NAACL Findings 2025
5. Memorization Without Overfitting — Tirumala et al., NeurIPS 2022
