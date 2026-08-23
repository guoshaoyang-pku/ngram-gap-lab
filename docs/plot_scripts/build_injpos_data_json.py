#!/usr/bin/env python3
"""Build injpos_ablation_data.json for the blog (v10 · 2000 steps).

Reads data/runs/nglab_{v,y,input}/ (train_log.jsonl, table_norm.jsonl,
summary.json) and writes the per-step gap + table RMS payload consumed by
the blog's index.html. Validation points land every 10 steps.
"""
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS_DIR = os.path.join(REPO_ROOT, "data", "runs_fixed")
OUT_PATH = os.environ.get("NGRAM_GAP_BLOG_DATA_JSON",
                         os.path.join(REPO_ROOT, "data", "injpos_ablation_data.json"))

RUN_IDS = {"v": "nglab1x_v10_v_fixed", "y": "nglab1x_v10_y_fixed", "input": "nglab1x_v10_input_fixed"}
RMS_KEY = "bigram.layer_01.table_0.rms"


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
    data = {}
    steps = None
    for key, run_id in RUN_IDS.items():
        run_dir = os.path.join(RUNS_DIR, run_id)
        with open(os.path.join(run_dir, "summary.json")) as f:
            summary = json.load(f)
        train_log = load_jsonl(os.path.join(run_dir, "train_log.jsonl"))
        table_norm = load_jsonl(os.path.join(run_dir, "table_norm.jsonl"))
        if not train_log:
            raise SystemExit(f"[injpos] no train_log for {run_id} at {run_dir}")
        data[key] = {
            "final_gap": summary["final_gap"],
            "final_train_loss": summary["final_train_loss"],
            "final_val_loss": summary["final_val_loss"],
            "gaps": {str(p["step"]): p["gap"] for p in train_log},
            "table_rms": {str(p["step"]): p.get(RMS_KEY, 0) for p in table_norm},
        }
        if steps is None:
            steps = [p["step"] for p in train_log]
    data["steps"] = steps
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(data, f, indent=1)
    for key in RUN_IDS:
        d = data[key]
        print(f"[injpos] {key}: final_gap={d['final_gap']:.4f} "
              f"points={len(d['gaps'])} steps={steps[0]}..{steps[-1]}")
    print(f"[injpos] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
