"""
visualize_rq3.py — Figure 6: LiLIS-K performance while varying datasets
Output: rq3_figure6.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Data
query_types = ['Point Query', 'Range Query', 'kNN Query']

paper = {
    'CHI': [607.05, 710.22, 743.74],
    'NYC': [ 82.59, 340.57, 650.20],
    'SYN': [141.76, 130.67, 185.50],
}
group = {
    'CHI': [ 145.5,  142.1,  282.3],
    'NYC': [ 780.3,  743.1,  565.5],
    'SYN': [2724.6, 2604.7, 2740.5],
}

colors_paper = {'CHI':'#E24B4A','NYC':'#378ADD','SYN':'#639922'}
colors_group = {'CHI':'#F09595','NYC':'#85B7EB','SYN':'#C0DD97'}
datasets = ['CHI','NYC','SYN']

x = np.arange(len(query_types))
w = 0.13
offsets = {'CHI':-2,'NYC':0,'SYN':2}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle('Hình 6: Hiệu năng LiLIS-K trên 3 tập dữ liệu (CHI / NYC / SYN)',
             fontsize=13, fontweight='bold', y=1.02)

for ax, (title, data) in zip(axes,
        [('Paper gốc (NYC 300M)', paper), ('Nhóm (NYC 46.5M, cluster 3 nodes)', group)]):
    for ds in datasets:
        pos = x + offsets[ds] * w
        bars = ax.bar(pos, data[ds], width=w*1.8,
                      color=colors_paper[ds] if data is paper else colors_group[ds],
                      edgecolor='white', linewidth=0.5, label=ds, zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+15,
                    f'{h:.0f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold',
                    color='#333')

    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(query_types, fontsize=10)
    ax.set_ylabel('Thời gian (ms)', fontsize=10)
    ax.set_ylim(0, max(max(v) for v in data.values()) * 1.35)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v,_: f'{v/1000:.1f}k' if v>=1000 else f'{int(v)}'))
    ax.grid(axis='y', linestyle='--', alpha=0.35, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

patches = [mpatches.Patch(color=colors_paper[ds], label=ds) for ds in datasets]
fig.legend(handles=patches, loc='lower center', ncol=3,
           fontsize=10, bbox_to_anchor=(0.5,-0.06), framealpha=0.9)

fig.text(0.5,-0.13,
    'Paper: SYN nhanh nhất (phân bố đều, dataset lớn) | '
    'Nhóm: CHI nhanh nhất (dataset nhỏ nhất 7.7M, Spark overhead dominate)',
    ha='center', fontsize=9, style='italic', color='#555')

plt.tight_layout()
plt.savefig('rq3_figure6.png', dpi=150,
            bbox_inches='tight', facecolor='white')
print("Saved: rq3_figure6.png")
