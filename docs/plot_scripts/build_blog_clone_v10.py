#!/usr/bin/env python3
"""Assemble the v10 blog clone (validation every 10 steps · 2000 steps).

Usage: python3 /tmp/build_blog_clone_v10.py [injpos_ablation_data.json]
"""
import json, os, re, shutil, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BLOG_ROOT = Path(os.environ.get("NGLAB_BLOG_ROOT", REPO_ROOT.parent / "guoshaoyang-pku.github.io"))
ORIG = Path(os.environ.get("NGLAB_BLOG_SOURCE", BLOG_ROOT / "blogs" / "ngram-gap-mechanism-guide"))
CLONE = Path(os.environ.get("NGLAB_BLOG_CLONE", BLOG_ROOT / "blogs" / "ngram-gap-mechanism-guide-v10"))
RUNS = Path(os.environ.get("NGLAB_RUNS_DIR", REPO_ROOT / "data" / "runs_fixed"))
FIGS = Path(os.environ.get("NGLAB_FIG_DIR", REPO_ROOT / "docs" / "figs" / "main"))
NEW_JSON = Path(sys.argv[1]) if len(sys.argv) > 1 else CLONE / "injpos_ablation_data.json"

def load_jsonl(path):
    pts = []
    if not os.path.exists(path):
        return pts
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pts.append(json.loads(line))
    return pts

def main():
    if os.path.exists(CLONE):
        shutil.rmtree(CLONE)
    shutil.copytree(ORIG, CLONE, ignore=shutil.ignore_patterns(".DS_Store"))

    # 1) replace static svgs with v10 versions
    for name in ["fig_gap.svg", "fig_loss.svg", "fig_table_norm.svg",
                 "fig_input_alignment.svg", "fig_freq_bigram.svg",
                 "fig_freq_trigram.svg", "fig_hitcount_dist.svg"]:
        src = os.path.join(FIGS, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(CLONE, name))

    # 2) interactive html figures (with log/logx naming mapping)
    #    v10 "fig_gap_vs_frequency_log.html"   = x-log / y-linear -> blog logx
    #    v10 "fig_gap_vs_frequency_loglog.html" = x-log / y-log    -> blog log
    for name in ["fig_gap_loss.html", "fig_loss_norm.html",
                 "fig_gap_by_freq.html", "fig_hitcount_dist.html"]:
        src = os.path.join(FIGS, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(CLONE, name))
    # current blog structure: loglog card -> fig_gap_vs_frequency_loglog.html,
    # log-x card -> fig_gap_vs_frequency_log.html (both direct copies)
    for name in ["fig_gap_vs_frequency_loglog.html", "fig_gap_vs_frequency_log.html"]:
        src = os.path.join(FIGS, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(CLONE, name))
    stale = os.path.join(CLONE, "fig_gap_vs_frequency_logx.html")
    if os.path.exists(stale):
        os.remove(stale)

    # 3) data json
    shutil.copy(NEW_JSON, os.path.join(CLONE, "injpos_ablation_data.json"))
    data = json.load(open(os.path.join(CLONE, "injpos_ablation_data.json")))

    # 4) epoch boundaries from the v10 input run
    train_log = load_jsonl(os.path.join(RUNS, "nglab1x_v10_input", "train_log.jsonl"))
    boundaries = []
    prev = None
    for p in train_log:
        if p.get("epoch") is not None and p["epoch"] != prev:
            if prev is not None:
                boundaries.append(int(p["step"]))
            prev = p["epoch"]
    if boundaries:
        epoch_len = round(sum(b2 - b1 for b1, b2 in zip(boundaries, boundaries[1:]))
                          / max(1, len(boundaries) - 1))
    else:
        epoch_len = 230
    epoch_steps = ", ".join(str(s) for s in boundaries)

    # 5) patch index.html
    html = open(os.path.join(ORIG, "index.html")).read()
    html = html.replace("seed 42 · 1000 steps", "seed 42 · 2000 steps")
    html = html.replace(
        "<b>seed 42 · 2000 steps</b></div></div>",
        "<b>seed 42 · 2000 steps</b></div>"
        "<div class=\"setting-item\"><small>validation</small><b>每 10 步</b></div>"
        "<div class=\"setting-item\"><small>freq eval</small><b>每 10 步</b></div></div>",
        1,
    )
    html = html.replace("gap @ 1000 steps", "gap @ 2000 steps")

    foot_old = "训练 shard 约 337 steps / epoch（2000 步约 6 个 epoch）；图中的竖线对应各 epoch 边界。"
    foot_new = (f"训练 shard 约 {epoch_len} steps / epoch（含 freq eval 消耗的 train batches；"
                f"虚线为数据自动标注的 epoch 边界：{epoch_steps}）。"
                f"validation 与 freq-bin eval 均每 10 步一次。")
    assert foot_old in html, "setting-foot anchor not found"
    html = html.replace(foot_old, foot_new)

    gaps = {k: data[k]["final_gap"] for k in ["v", "y", "input"]}
    for k in ["v", "y", "input"]:
        html = re.sub(rf'(<div class="value" id="card-{k}">)[^<]*(</div>)',
                      rf"\g<1>{gaps[k]:.2f}\g<2>", html)
    nogram = json.load(open(os.path.join(RUNS, "nglab1x_v10_nogram", "summary.json")))
    nogram_gap = nogram["final_gap"]
    html = re.sub(r'(<div class="metric base"><div class="label">无 n-gram</div><div class="value">)[^<]*(</div>)',
                  rf"\g<1>{nogram_gap:.2f}\g<2>", html)

    marker = ("<strong>v10 克隆</strong>：本页为 validation 每 10 步 · 2000 steps 的重跑版本"
              "（原版为 val 每 50 步 · 1000 steps，见 <a href=\"../ngram-gap-mechanism-guide/index.html\">原版页面</a>）。")
    anchor = '<div class="archive"><strong>版本说明</strong>'
    assert anchor in html, "archive anchor not found"
    html = html.replace(anchor, '<div class="archive">' + marker + '<br><strong>版本说明</strong>', 1)

    open(os.path.join(CLONE, "index.html"), "w").write(html)
    print(f"[clone] assembled {CLONE}")
    print(f"[clone] final gaps v={gaps['v']:.4f} y={gaps['y']:.4f} input={gaps['input']:.4f} nogram={nogram_gap:.4f}")
    print(f"[clone] epoch_len={epoch_len} boundaries={boundaries}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
