"""
visualize_summary.py — Tổng kết Speedup LiLIS-K vs Sedona
Output: summary_takeaways.png
"""
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle('Tổng kết Speedup LiLIS-K so với Sedona\n(Cluster 3 nodes LAN)',
             fontsize=13, fontweight='bold', y=1.02)

methods       = ['vs Sedona-N\nCHI kNN',
                 'vs Sedona-N\nNYC kNN',
                 'vs Sedona-RK\nNYC kNN',
                 'vs Sedona-QQ\nNYC Range']
speedups      = [26.0, 10.0, 22.4, 13.3]
colors_bar    = ['#1D9E75', '#1D9E75', '#378ADD', '#9B59B6']
dataset_label = ['CHI', 'NYC', 'NYC', 'NYC']

bars = ax.bar(range(len(methods)), speedups,
              color=colors_bar, alpha=0.88,
              edgecolor='white', linewidth=0.8, zorder=3)

# Nhãn giá trị
for bar, val in zip(bars, speedups):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f'{val:.0f}x', ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='#333')

# Ghi chú dataset
for i, ds in enumerate(dataset_label):
    ax.text(i, 1.0, ds, ha='center', va='bottom',
            fontsize=8, color='white', fontweight='bold', zorder=4)

ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, fontsize=9.5)
ax.set_ylabel('Speedup (x)', fontsize=11)
ax.set_ylim(0, 33)
ax.set_yticks([0, 5, 10, 15, 20, 25, 30])
ax.grid(axis='y', linestyle='--', alpha=0.35, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Caption giải thích
ax.text(1.5, 29.5,
        'Speedup nho hon paper (100-1000x) do dataset\n'
        'nho hon 6.5x va cluster 3 nodes thay vi 7 nodes',
        ha='center', fontsize=8.5, style='italic', color='#666',
        bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow',
                  ec='#cccc88', alpha=0.9))

plt.tight_layout()
plt.savefig('summary_takeaways.png', dpi=150,
            bbox_inches='tight', facecolor='white')
print("Saved: summary_takeaways.png")
