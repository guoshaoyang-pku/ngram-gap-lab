import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open('freq_bin_loss.jsonl') as f:
    fb = [json.loads(l) for l in f if l.strip()]

last = fb[-1]
bucket_order = ["novel","1","2","3","4","5","6-10","11-20","21-50",
                "51-100","101-200","201-500","501-1k","1k-5k","5k+"]
bucket_colors = {"novel":"#E91E63","1":"#F44336","2":"#FF5722","3":"#FF9800",
                 "4":"#FFC107","5":"#FFEB3B","6-10":"#CDDC39","11-20":"#8BC34A",
                 "21-50":"#4CAF50","51-100":"#009688","101-200":"#00BCD4",
                 "201-500":"#03A9F4","501-1k":"#2196F3","1k-5k":"#3F51B5","5k+":"#673AB7"}

def plot_branch(branch_key, branch_name):
    values = {}
    for b in bucket_order:
        td = last['train'][branch_key].get(b, {"token_count":0,"frac":0,"mean_loss":0,"total_contrib":0})
        vd = last['val'][branch_key].get(b, {"token_count":0,"frac":0,"mean_loss":0,"total_contrib":0})
        values[b] = {
            "train_loss": td.get("mean_loss", 0),
            "val_loss": vd.get("mean_loss", 0),
            "gap": vd.get("mean_loss", 0) - td.get("mean_loss", 0),
            "train_contrib": td.get("total_contrib", 0),
            "val_contrib": vd.get("total_contrib", 0),
            "val_frac": vd.get("frac", 0),
        }

    pos = np.arange(len(bucket_order))
    w = 0.38

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, facecolor='white')

    # Panel 1: Per-bucket mean loss
    axes[0].bar(pos - w/2, [values[b]["train_loss"] for b in bucket_order], w,
                color='#2d6f9f', alpha=0.8, label='Train mean loss')
    axes[0].bar(pos + w/2, [values[b]["val_loss"] for b in bucket_order], w,
                color='#c4493d', alpha=0.8, label='Val mean loss')
    axes[0].set_ylabel('Mean loss')
    axes[0].set_title(f'{branch_name.capitalize()} context — Per-bucket loss (step {last["step"]})\n'
                      f'Bigram-only model · Markov λ=0.8 · Train=1shard · 3000steps · gap={last.get("val_loss",0)-last.get("train_loss",0):+.2f}',
                      fontweight='bold')
    axes[0].legend(loc='upper left', fontsize=8)
    axes[0].grid(alpha=0.3, axis='y')

    # Panel 2: Per-bucket gap with fraction labels
    gaps = [values[b]["gap"] for b in bucket_order]
    axes[1].bar(pos, gaps, color=[bucket_colors[b] for b in bucket_order], alpha=0.9)
    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Gap (val - train)')
    axes[1].grid(alpha=0.3, axis='y')
    for i, b in enumerate(bucket_order):
        frac = values[b]["val_frac"]
        if frac > 0.01:
            axes[1].text(i, gaps[i] + (0.05 if gaps[i] >= 0 else -0.1),
                        f'{frac*100:.1f}%', ha='center', fontsize=7, color='gray')

    # Panel 3: Gap contribution (frac × gap)
    axes[2].bar(pos - w/2, [values[b]["train_contrib"] for b in bucket_order], w,
                color='#2d6f9f', alpha=0.8, label='Train contrib')
    axes[2].bar(pos + w/2, [values[b]["val_contrib"] for b in bucket_order], w,
                color='#c4493d', alpha=0.8, label='Val contrib')
    axes[2].set_ylabel('Total contribution')
    axes[2].legend(loc='upper left', fontsize=8)
    axes[2].grid(alpha=0.3, axis='y')
    axes[2].set_xticks(pos)
    axes[2].set_xticklabels(bucket_order, rotation=45, ha='right')
    axes[2].set_xlabel('Training hit-count bucket')

    plt.tight_layout()
    out = f'freq_{branch_key}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    return out

plot_branch('bigram', 'bigram')
plot_branch('trigram', 'trigram')
print('done')
