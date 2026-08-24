#!/usr/bin/env python3
"""build_report.py — 把 report.md 的核心内容渲染成独立的 report.html

- 图片以 SVG 内联方式嵌入（不依赖外部文件），可直接用浏览器打开。
- 表格与列表从本文件里的结构化数据生成（不走 markdown 解析，避免引入依赖）。
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
OUT = HERE / "report.html"

SVG_ORDER = [
    ("fig_beta2_curves.svg", "β₂ 消融 · gap 全程曲线簇"),
    ("fig_lr_curves.svg", "表学习率消融 · gap 全程曲线簇（β₂=0.999）"),
    ("fig_val_loss_beta2_lr.svg", "val loss 体检 · β₂=0.99 在高表学习率下是否崩坏"),
    ("fig_val_loss_short_epoch.svg", "短 epoch 家族 val loss（0.25x / 0.5x，表学习率 ×1）"),
    ("fig_beta2_finalgap.svg", "β₂ 消融 · 终点 gap 汇总"),
    ("fig_beta2_spread_vs_lr.svg", "关键分析 · β₂ 压差随表学习率衰减"),
    ("fig_lr_sweep_b2_099.svg", "β₂=0.99 下的表学习率扫描（补点完成后完整）"),
]

FINDINGS = [
    ("F1", "β₂ 越小、gap 越大（方向确定）",
     "16 个 β₂ 专题 run 一致：β₂ 降低 → 终点 gap 增大。例 2-epoch · LR×2：β₂=0.98 → 2.10，"
     "β₂=0.99999 → 0.72（差 3 倍）。机制：β₂ 控制梯度平方 EMA 窗口，窗口短 → 稀疏低频行"
     "被更激进更新 → 记忆更快 → gap 更大。β₂ 本质是低频行的学习率杠杆。"),
    ("F2", "表学习率越大、gap 越大，但伴随训练崩坏（用户怀疑被证实）",
     "1x shard · β₂=0.99：×1 健康（val 最低 4.49 → 结束 4.61）；×2 崩坏（val 涨到 7.05，"
     "train 塌到 1.55）；×4 更糟（val 峰值 10.8）。2-epoch 同模式。表学习率 ×2 以上 = "
     "表写入过猛 = 过拟合。这些高 LR run 的大 gap 不能用于 β₂ 比较。"),
    ("F3", "修正一个表面结论（关键）",
     "曲线簇在表学习率 ×4 时合并，曾被误读为「β₂ 在高表学习率下不重要」。正确解读：×4 "
     "把训练搞崩了，所有 β₂ 一起烂，β₂ 差异被训练崩坏淹没。压差数据：1x·LR×2 +1.01(+22%) "
     "→ 1x·LR×4 只剩 +0.34(+6%)；2ep·LR×2 +0.36(+21%) → 2ep·LR×4 +0.12(≈0)。"
     "不是 β₂ 不重要，而是高表学习率实验不可信。表学习率 ×1 是唯一健康设置。"),
    ("F4", "推论 · 主线 β₂ 默认值的选择",
     "主线表学习率 = ×1（唯一健康）。β₂=0.999(默认) 与 β₂=0.99 的主线对照正在补跑。"
     "短 epoch 侧面证据（×1）：0.25x 9.69 vs 8.97、0.5x 3.99 vs 4.02 —— 差异很小。"
     "待补点完成后下结论；若差异 <20% 就保持默认 0.999。"),
]

TABLE_ROWS = [
    ("1x·LR×2", "b2_098 / b2_099 / default", "0.98 / 0.99 / 0.999", "×2",
     "5.68 / 5.50 / 4.67", "⚠️"),
    ("1x·LR×4", "b2_{098,099}_fixed + 4x_fixed", "0.98 / 0.99 / 0.999", "×4",
     "5.60 / 5.50 / 5.26", "⚠️ 峰值 10.7"),
    ("2ep·LR×2", "2x_b2_{098,099} 等 5 点", "0.98 → 0.99999", "×2",
     "2.10 → 0.72", "✅"),
    ("2ep·LR×4", "4x_b2_*_fixed", "0.98 → 0.9999", "×4",
     "2.22 → 2.10", "⚠️ 峰值 10.7"),
    ("基线", "input_v10 / input_v10_fv / nogram", "0.999", "×1",
     "1.87 / 0.58 / 0.25", "✅"),
    ("短epoch", "025x / 05x b2_099 + 0.999", "0.99 / 0.999", "×1",
     "见 F4", "✅"),
]

NEW_RUNS = [
    ("nglab1x_opt_rmsprop_b2_099_lr1", "β₂=0.99 · 表 LR×1 · 1x shard · 2000 步", "GPU5"),
    ("nglab2x_opt_rmsprop_b2_099_lr1", "β₂=0.99 · 表 LR×1 · 2-epoch shard · 2000 步", "GPU7"),
]


def svg_block(name: str, caption: str) -> str:
    p = FIGS / name
    if not p.exists():
        return f'<div class="fig"><p class="missing">（{name} 尚未生成）</p></div>'
    svg = p.read_text()
    return (f'<figure class="fig"><div class="svgbox">{svg}</div>'
            f'<figcaption>{caption}</figcaption></figure>')


def main() -> None:
    findings_html = "".join(
        f'<div class="finding"><div class="fid">{fid}</div>'
        f'<div><h3>{title}</h3><p>{body}</p></div></div>'
        for fid, title, body in FINDINGS)

    rows_html = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
        for row in TABLE_ROWS)

    new_runs_html = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
        for row in NEW_RUNS)

    figs_html = "".join(svg_block(n, c) for n, c in SVG_ORDER)

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>附录 · 表学习率 × β₂ 消融</title>
<style>
  body {{ font-family: -apple-system, "PingFang SC", sans-serif; max-width: 1100px;
          margin: 0 auto; padding: 24px; color: #222; line-height: 1.55; }}
  h1 {{ font-size: 26px; border-bottom: 2px solid #4878CF; padding-bottom: 8px; }}
  h2 {{ font-size: 20px; margin-top: 34px; color: #2c5fa0; }}
  h3 {{ font-size: 16px; margin: 8px 0 4px; }}
  .meta {{ color: #666; font-size: 14px; background: #f5f8fc; padding: 12px 16px;
           border-radius: 8px; margin: 14px 0; }}
  .finding {{ display: flex; gap: 14px; background: #fafbfc; border-left: 4px solid #4878CF;
              padding: 12px 16px; border-radius: 6px; margin: 14px 0; }}
  .fid {{ font-weight: 700; color: #4878CF; font-size: 18px; min-width: 40px; }}
  .fig {{ margin: 22px 0; }}
  .fig figcaption {{ font-size: 13px; color: #555; text-align: center; margin-top: 4px; }}
  .svgbox svg {{ width: 100%; height: auto; max-width: 1080px; display: block; margin: auto; }}
  table {{ border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 14px; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
  th {{ background: #eef3fa; }}
  code {{ background: #f0f2f5; padding: 1px 5px; border-radius: 4px; font-size: 13px; }}
  .missing {{ color: #b00; font-size: 14px; }}
</style>
</head>
<body>
<h1>附录 · 表学习率 × β₂ 消融</h1>
<div class="meta">
实验线：T-table-opt（表优化器消融）子专题 · 状态：🟡 进行中（两个补点运行中）<br>
数据源：<code>data/runs_fixed/*_fixed/</code>（post-fix 权威数据） · 代码：<code>extract_data.py</code> + <code>make_figures.py</code>
</div>

<h2>1. 核心发现</h2>
{findings_html}

<h2>2. 全部图片</h2>
{figs_html}

<h2>3. 已有实验（_fixed 权威数据）</h2>
<table>
<tr><th>家族</th><th>run</th><th>β₂</th><th>表 LR</th><th>终点 gap</th><th>val 健康</th></tr>
{rows_html}
</table>

<h2>4. 本附录新跑（2026-08-24）</h2>
<table>
<tr><th>run</th><th>设置</th><th>GPU</th><th>状态</th></tr>
{new_runs_html}<tr><td colspan="4">两个补点 🏃 运行中（10:36 启动，~50 min）</td></tr>
</table>

<h2>5. 对主线的行动建议</h2>
<ol>
<li><b>表学习率保持 ×1</b>（0.004）。×2/×4 是训练崩坏设置，未来不再使用；基于它们的 β₂ 读数按 F3 重新解释。</li>
<li>β₂ 默认值待补点：若 β₂=0.99 vs 0.999（主线配置）差异 &lt;20%，<b>保持默认 0.999</b>；明显更小且无副作用才切换。</li>
<li><code>experiment-log.md</code> §9c/§9d 的 β₂ 段落，待补点结论后重写（plan-3 T2 的输入）。</li>
</ol>

</body>
</html>
"""
    OUT.write_text(html)
    print(f"wrote {OUT}  ({len(html)} chars)")


if __name__ == "__main__":
    main()
