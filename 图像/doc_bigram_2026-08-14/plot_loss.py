import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open('train_log.jsonl') as f:
    data = [json.loads(l) for l in f if l.strip()]

steps = [d['step'] for d in data]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, facecolor='white')

ax1.plot(steps, [d['train_loss'] for d in data], '#2d6f9f', lw=0.6, alpha=0.6, label='Train loss')
ax1.plot(steps, [d['val_loss'] for d in data], '#c4493d', lw=1.5, label='Val loss')
prev = 0
for d in data:
    if d['epoch'] > prev:
        for ax in (ax1, ax2):
            ax.axvline(d['step'], color='gray', ls=':', alpha=0.5)
        prev = d['epoch']
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_title(f'Shared-dict data · vocab=256 · content_prob=0.1 · steps {steps[0]}–{steps[-1]}\n'
              f'Train={data[-1]["train_loss"]:.4f}  Val={data[-1]["val_loss"]:.4f}  '
              f'Gap={data[-1]["gap"]:+.4f}', fontweight='bold')

ax2.plot(steps, [d['gap'] for d in data], 'purple', lw=1.2)
ax2.axhline(0, color='gray', ls='--', alpha=0.4)
ax2.set_xlabel('Step')
ax2.set_ylabel('Gap')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('loss_gap.png', dpi=150, bbox_inches='tight')

# Per-epoch summary
print('Per-epoch avg:')
for ep in sorted(set(d['epoch'] for d in data)):
    g = [d['gap'] for d in data if d['epoch'] == ep]
    t = [d['train_loss'] for d in data if d['epoch'] == ep]
    v = [d['val_loss'] for d in data if d['epoch'] == ep]
    print(f'  Epoch {ep}: train={np.mean(t):.4f}  val={np.mean(v):.4f}  gap={np.mean(g):+.4f}')
