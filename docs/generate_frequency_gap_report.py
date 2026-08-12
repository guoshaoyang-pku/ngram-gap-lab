#!/usr/bin/env python3
"""Build the blog-style hit-count loss / contribution / gap report for one run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BUCKETS = [
    "novel", "1", "2", "3", "4", "5", "6-10", "11-20", "21-50",
    "51-100", "101-200", "201-500", "501-1k", "1k-5k", "5k+",
]
COLORS = [
    "#E91E63", "#F44336", "#FF5722", "#FF9800", "#FFC107", "#FFEB3B",
    "#CDDC39", "#8BC34A", "#4CAF50", "#009688", "#00BCD4", "#03A9F4",
    "#2196F3", "#3F51B5", "#673AB7",
]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def series(records: list[dict], train_key: str, val_key: str) -> dict:
    out = {branch: {bucket: {
        "steps": [], "epochs": [], "reasons": [], "train_loss": [], "val_loss": [],
        "train_frac": [], "val_frac": [], "train_contribution": [],
        "val_contribution": [], "gap_contribution": [], "mean_loss_gap": [],
    } for bucket in BUCKETS} for branch in ("bigram", "trigram")}
    for row in records:
        for branch in out:
            for bucket in BUCKETS:
                train = row[train_key][branch][bucket]
                val = row[val_key][branch][bucket]
                dest = out[branch][bucket]
                dest["steps"].append(row["step"])
                dest["epochs"].append(row["epoch"])
                dest["reasons"].append(row["reason"])
                dest["train_loss"].append(train["mean_loss"])
                dest["val_loss"].append(val["mean_loss"])
                dest["train_frac"].append(train["frac"])
                dest["val_frac"].append(val["frac"])
                dest["train_contribution"].append(train["total_contrib"])
                dest["val_contribution"].append(val["total_contrib"])
                dest["gap_contribution"].append(
                    row["gap_contribution"][branch][bucket]["contribution"])
                dest["mean_loss_gap"].append(
                    row["gap_contribution"][branch][bucket]["mean_loss_gap"])
    return out


def fixed_gram_series(records: list[dict]) -> dict:
    """Normalize fixed gram records to the same plotting shape as old sources."""
    out = {branch: {bucket: {
        "steps": [], "epochs": [], "reasons": [], "train_loss": [], "val_loss": [],
        "train_sample_count": [], "val_sample_count": [], "gap_contribution": [],
        "mean_loss_gap": [],
    } for bucket in BUCKETS} for branch in ("bigram", "trigram")}
    for row in records:
        for branch in out:
            for bucket in BUCKETS:
                stats = row.get("branches", {}).get(branch, {}).get(bucket, {})
                dest = out[branch][bucket]
                dest["steps"].append(row.get("step"))
                dest["epochs"].append(row.get("epoch"))
                dest["reasons"].append(row.get("reason", ""))
                dest["train_loss"].append(stats.get("train_mean_loss"))
                dest["val_loss"].append(stats.get("val_mean_loss"))
                dest["train_sample_count"].append(stats.get("train_sample_count", 0))
                dest["val_sample_count"].append(stats.get("val_sample_count", 0))
                gap = stats.get("gap_contribution")
                dest["gap_contribution"].append(gap)
                dest["mean_loss_gap"].append(gap)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="data/runs/nglab_baseline_input_fixed_gram")
    parser.add_argument("--out", default="docs/frequency-gap-by-hit-count.html")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    meta_path = run_dir / "frequency_measurement_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    online = read_jsonl(run_dir / "online_frequency_gap_contribution.jsonl")
    fixed = read_jsonl(run_dir / "fixed_probe_frequency_gap_contribution.jsonl")
    fixed_gram = read_jsonl(run_dir / "fixed_gram_frequency_gap_contribution.jsonl")
    payload = {
        "meta": meta,
        "onlineLoss": read_jsonl(run_dir / "online_loss.jsonl"),
        "validation": read_jsonl(run_dir / "train_log.jsonl"),
        "frequency": {
            "online": series(online, "train_writer", "online_val") if online else {},
            "fixed": series(fixed, "train_probe", "val_probe") if fixed else {},
            "fixed_gram": fixed_gram_series(fixed_gram) if fixed_gram else {},
        },
        "bucketBasis": "train-frequency index",
        "fixedReads": [{"step": x["step"], "epoch": x["epoch"], "reason": x["reason"]}
                       for x in fixed],
    }
    html = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>按 hit count 分桶的 loss / gap</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 16px; background: #fafafa; color: #202124; }
h1 { font-size: 1.35rem; margin-bottom: .25rem; }
p { max-width: 1050px; line-height: 1.5; }
.controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: end; margin: 16px 0 8px; }
.controls label { display: grid; gap: 4px; font-size: .9rem; }
button, select { font: inherit; padding: 4px 8px; border: 1px solid #b7b7b7; border-radius: 4px; background: white; }
button.active { background: #30343b; color: white; }
.chart { width: 100%; max-width: 1250px; height: 450px; }
.note { font-size: .88rem; color: #555; }
</style>
</head>
<body>
<h1>按 hit count 分桶的 loss / gap</h1>
<p class="note">每条彩色曲线对应一个按 train corpus hit count 分桶的 bucket。<b>per-token loss</b> 是该桶 token loss 的平均；<b>mean per-token gap</b> 是同一 train-frequency bucket 中 val 平均 token loss − train 平均 token loss，不乘 token fraction。<b>fixed gram sample</b> 是训练前按 source、bigram/trigram 和 bucket 固定抽取的 token occurrence 集合；它与 fixed batch probe 独立。online 是当前 writer batch 与独立 validation batch，fixed batch probe 是固定连续 batch 对照。</p>
<div class="controls">
  <label>上图取点间隔<select id="lossStride"><option value="1">每 1 step</option><option value="5">每 5 step</option><option value="10" selected>每 10 step</option><option value="25">每 25 step</option><option value="50">每 50 step</option></select></label>
  <label>统计来源<select id="source"><option value="fixed_gram" selected>fixed gram sample</option><option value="online">online writer batch</option><option value="fixed">fixed batch probe</option></select></label>
  <label>context<select id="branch"><option value="bigram">bigram</option><option value="trigram">trigram</option></select></label>
  <span><button type="button" class="metric" data-metric="loss">per-token loss</button><button type="button" class="metric active" data-metric="gap">mean per-token gap</button></span>
</div>
<div id="lossChart" class="chart"></div>
<div id="frequencyChart" class="chart"></div>
<p id="status" class="note"></p>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>
const report = __DATA__;
const buckets = __BUCKETS__;
const colors = __COLORS__;
let metric = "gap";

function epochShapes() {
  const n = report.meta?.sampling?.estimated_steps_per_epoch || 0;
  const maxStep = Math.max(...report.onlineLoss.map(x => x.step), 0);
  const shapes = [];
  for (let step = n; step < maxStep; step += n) {
    shapes.push({type:"line", x0:step, x1:step, y0:0, y1:1, yref:"paper", line:{color:"#aaa",dash:"dot",width:1}});
  }
  return shapes;
}
function fixedProbeLocationShapes() {
  const epochSteps = report.meta?.sampling?.estimated_steps_per_epoch || 0;
  const probeBatches = report.meta?.fixed_probe?.train_batches || 0;
  const gradAccum = report.meta?.geometry?.grad_accum || 1;
  const probeOffset = report.meta?.fixed_probe?.train_offset_optimizer_steps || 0;
  const maxStep = Math.max(...report.onlineLoss.map(x => x.step), 0);
  if (!epochSteps || !probeBatches) return [];

  const probeSteps = Math.ceil(probeBatches / gradAccum);
  const shapes = [];
  for (let start = 1; start <= maxStep; start += epochSteps) {
    shapes.push({
      type: "rect", x0: start + probeOffset - .5,
      x1: Math.min(maxStep, start + probeOffset + probeSteps - 1) + .5,
      y0: 0, y1: 1, yref: "paper", layer: "below",
      fillcolor: "rgba(33, 150, 243, 0.12)", line: {width: 0}
    });
  }
  return shapes;
}
function thin(points, stride) {
  return points.filter(p => p.step % stride === 0 || p.step === points.at(-1)?.step);
}
function plotLoss() {
  const stride = Number(document.getElementById("lossStride").value);
  const train = thin(report.onlineLoss, stride);
  const val = report.validation;
  Plotly.react("lossChart", [
    {x:train.map(x=>x.step), y:train.map(x=>x.train_writer_loss), mode:"lines", name:"writer train loss", line:{color:"#2962ff",width:1.4},
      hovertemplate:"step=%{x}<br>writer train loss=%{y:.5f}<extra></extra>"},
    {x:val.map(x=>x.step), y:val.map(x=>x.val_loss), mode:"lines+markers", name:"validation loss", line:{color:"#d84315",width:2},
      hovertemplate:"step=%{x}<br>validation loss=%{y:.5f}<extra></extra>"}
  ], {title:"Online loss", xaxis:{title:"step"}, yaxis:{title:"cross-entropy loss"},
      shapes:epochShapes(), margin:{l:65,r:24,t:48,b:54}, legend:{x:.02,y:.98}});
}
function plotFrequency() {
  const source = document.getElementById("source").value;
  const branch = document.getElementById("branch").value;
  const data = report.frequency[source]?.[branch];
  const status = document.getElementById("status");
  if (!data) {
    Plotly.purge("frequencyChart");
    status.textContent = "当前运行尚未写入该统计。";
    return;
  }
  if (source === "fixed") {
    const offset = report.meta?.fixed_probe?.train_offset_optimizer_steps || 0;
    status.textContent = `bucket 使用 train corpus hit count；浅蓝竖带是固定连续 train probe 在每个 replay epoch 中的原始 writer-batch 位置（从 epoch 起点偏移 ${offset} 个 optimizer step），三角标记是评估该固定 probe 的 checkpoint。固定 val probe 取独立 validation iterator 的同样批数。它与 fixed gram sample 不是同一批样本。`;
  } else if (source === "fixed_gram") {
    status.textContent = "bucket 使用 train frequency index；每个 source、branch、bucket 的 occurrence 在训练前固定，gap = val mean token loss − train mean token loss，不乘 token fraction。每个点是一个 checkpoint。";
  } else {
    status.textContent = "bucket 使用 train corpus hit count；曲线点来自实际 writer batch 与独立 moving validation batch。密集点围绕 replay epoch 边界和固定 train probe 所在位置取得。";
  }
  const traces = [];
  buckets.forEach((bucket, i) => {
    const row = data[bucket];
    let trainY, valY, gapY;
    if (metric === "loss") {
      trainY = row.train_loss; valY = row.val_loss;
      gapY = row.val_loss.map((v, j) =>
        v == null || row.train_loss[j] == null ? null : v - row.train_loss[j]);
    } else {
      gapY = row.mean_loss_gap;
    }
    const common = {x:row.steps, customdata:row.steps.map((_,j)=>[
      row.epochs[j], gapY[j], row.reasons[j], row.train_loss[j], row.val_loss[j],
      row.train_sample_count?.[j] ?? null, row.val_sample_count?.[j] ?? null
    ]), line:{color:colors[i],width:1.8}};
    if (metric === "gap") {
      traces.push({...common, y:gapY, mode:"lines+markers", name:bucket,
        hovertemplate:"bucket="+bucket+"<br>step=%{x} (epoch %{customdata[0]})<br>mean per-token gap=%{y:.6f}<br>train mean loss=%{customdata[3]:.6f}<br>val mean loss=%{customdata[4]:.6f}<br>train samples=%{customdata[5]}<br>val samples=%{customdata[6]}<br>reason=%{customdata[2]}<extra></extra>"});
    } else {
      traces.push({...common, y:valY, mode:"lines+markers", name:bucket+" (val)",
        hovertemplate:"bucket="+bucket+" · val<br>step=%{x} (epoch %{customdata[0]})<br>per-token loss=%{y:.6f}<br>train mean loss=%{customdata[3]:.6f}<br>val mean loss=%{customdata[4]:.6f}<br>train samples=%{customdata[5]}<br>val samples=%{customdata[6]}<br>reason=%{customdata[2]}<extra></extra>"});
      traces.push({...common, y:trainY, mode:"lines", name:bucket+" (train)", showlegend:false,
        line:{color:colors[i],width:1,dash:"dash"},
        hovertemplate:"bucket="+bucket+" · train<br>step=%{x} (epoch %{customdata[0]})<br>per-token loss=%{y:.6f}<br>train samples=%{customdata[5]}<br>val samples=%{customdata[6]}<br>reason=%{customdata[2]}<extra></extra>"});
    }
  });
  if (source === "fixed") {
    const reads = report.fixedReads;
    traces.push({x:reads.map(x=>x.step), y:reads.map(()=>0), yaxis:"y2", mode:"markers", name:"fixed probe evaluation",
      marker:{symbol:"triangle-up",size:8,color:"#222"}, customdata:reads.map(x=>[x.epoch,x.reason]),
      hovertemplate:"fixed probe evaluation<br>step=%{x}<br>epoch=%{customdata[0]}<br>reason=%{customdata[1]}<extra></extra>"});
  }
  const sourceTitle = source === "online" ? "online moving batches" : source === "fixed" ? "fixed batch probe" : "fixed gram sample";
  const title = `${sourceTitle} · ${branch} · ${metric === "gap" ? "mean per-token gap (val − train)" : "train / val per-token loss"}`;
  const shapes = source === "fixed"
    ? [...epochShapes(), ...fixedProbeLocationShapes()]
    : epochShapes();
  Plotly.react("frequencyChart", traces, {title, xaxis:{title:"step"}, yaxis:{title:metric === "gap" ? "mean per-token loss gap" : "per-token loss", zeroline:metric==="gap"},
    yaxis2:{overlaying:"y",side:"right",range:[-1,1],visible:source==="fixed",showticklabels:false,title:source==="fixed" ? "probe reads" : ""},
    shapes, margin:{l:72,r:80,t:50,b:55}, legend:{x:.01,y:.99,font:{size:9}}});
}
document.getElementById("lossStride").addEventListener("change", plotLoss);
document.getElementById("source").addEventListener("change", plotFrequency);
document.getElementById("branch").addEventListener("change", plotFrequency);
document.querySelectorAll(".metric").forEach(btn => btn.addEventListener("click", () => {
  metric = btn.dataset.metric; document.querySelectorAll(".metric").forEach(x=>x.classList.toggle("active",x===btn)); plotFrequency();
}));
plotLoss(); plotFrequency();
</script>
</body></html>"""
    html = html.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    html = html.replace("__BUCKETS__", json.dumps(BUCKETS))
    html = html.replace("__COLORS__", json.dumps(COLORS))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"wrote {output} ({len(fixed_gram)} fixed gram / {len(online)} online / {len(fixed)} fixed-batch checkpoints)")


if __name__ == "__main__":
    main()
