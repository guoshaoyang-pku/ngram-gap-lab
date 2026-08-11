import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('train_log.jsonl') as f:
    data = [json.loads(l) for l in f if l.strip()]

steps = [d['step'] for d in data]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, facecolor='white')

ax1.plot(steps, [d['train_loss'] for d in data], '#2d6f9f', linewidth=1.0, alpha=0.6, label='Train loss')
ax1.plot(steps, [d['val_loss'] for d in data], '#c4493d', linewidth=1.5, label='Val loss')

prev_ep = 0
for d in data:
    if d['epoch'] > prev_ep:
        for ax in (ax1, ax2):
            ax.axvline(d['step'], color='gray', linestyle=':', alpha=0.35)
        prev_ep = d['epoch']

ax1.set_ylabel('Cross-entropy loss')
ax1.legend(loc='upper right')
ax1.set_title(f'Pure nanoGPT (baseline) — Markov λ=0.8 · 3000 steps\nFinal: train={data[-1]["train_loss"]:.4f}  val={data[-1]["val_loss"]:.4f}  gap={data[-1]["gap"]:+.4f}')
ax1.grid(alpha=0.3)

ax2.plot(steps, [d['gap'] for d in data], 'purple', linewidth=1.2)
ax2.axhline(0, color='gray', linestyle='--', alpha=0.4)
ax2.set_xlabel('Step')
ax2.set_ylabel('Gap (val - train)')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('loss.png', dpi=150, bbox_inches='tight')
