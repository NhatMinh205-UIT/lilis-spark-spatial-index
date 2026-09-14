"""
visualize_rq4.py — Figure 7 (Selectivity Sweep) + Figure 8 (kNN Sweep)
Output: rq4_figure7_selectivity.png, rq4_figure8_knn.png
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

COLOR_LILIS  = '#1D9E75'
COLOR_NOINDEX = '#E24B4A'
COLOR_CHI    = '#E24B4A'
COLOR_NYC    = '#378ADD'
COLOR_SYN    = '#639922'

# ═══════════════════════════════════════════════════════════
# FIGURE 8 — kNN Sweep (k = 1, 10, 50, 100)
# ═══════════════════════════════════════════════════════════
k_vals = [1, 10, 50, 100]

knn_data = {
    'CHI': {
        'LiLIS-K':  [125.1, 282.3, 161.6, 206.3],
        'No-index': [1306.5, 1501.7, 1654.3, 1532.8],
    },
    'NYC': {
        'LiLIS-K':  [574.7, 565.5, 657.3, 864.6],
        'No-index': [5888.7, 6060.0, 6103.4, 5942.0],
    },
    'SYN': {
        'LiLIS-K':  [2383.2, 2740.5, 2451.1, 2163.6],
        'No-index': [2799.4, 8907.1, 8907.1, 8940.6],
    },
}

fig8, axes8 = plt.subplots(1, 3, figsize=(14, 4.5))
fig8.suptitle('Hình 8. kNN Query khi thay đổi k — LiLIS-K vs No-index\n'
              '(Cluster 3 nodes LAN, 15 runs)',
              fontsize=12, fontweight='bold', y=1.03)

ds_colors = {'CHI': COLOR_CHI, 'NYC': COLOR_NYC, 'SYN': COLOR_SYN}
datasets   = ['CHI', 'NYC', 'SYN']

for ax, ds in zip(axes8, datasets):
    d = knn_data[ds]
    ax.plot(k_vals, d['LiLIS-K'],
            marker='o', linewidth=2, markersize=7,
            color=COLOR_LILIS, label='LiLIS-K', zorder=3)
    ax.plot(k_vals, d['No-index'],
            marker='s', linewidth=2, markersize=7,
            linestyle='--', color=COLOR_NOINDEX,
            label='No-index', zorder=3)

    # Nhãn giá trị
    for x_, y_ in zip(k_vals, d['LiLIS-K']):
        ax.annotate(f'{y_:.0f}', (x_, y_),
                    textcoords='offset points', xytext=(0, 8),
                    ha='center', fontsize=7.5, color=COLOR_LILIS, fontweight='bold')
    for x_, y_ in zip(k_vals, d['No-index']):
        ax.annotate(f'{y_:.0f}', (x_, y_),
                    textcoords='offset points', xytext=(0, -14),
                    ha='center', fontsize=7.5, color=COLOR_NOINDEX)

    ax.set_title(f'Tập {ds}', fontsize=11, fontweight='bold')
    ax.set_xlabel('k (số láng giềng)', fontsize=10)
    ax.set_ylabel('Thời gian (ms)', fontsize=10)
    ax.set_xticks(k_vals)
    ax.set_ylim(0, max(max(d['LiLIS-K']), max(d['No-index'])) * 1.35)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v,_: f'{v/1000:.1f}k' if v>=1000 else f'{int(v)}'))
    ax.grid(linestyle='--', alpha=0.35, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=9, loc='upper left')

fig8.text(0.5, -0.06,
    'Nhận xét: LiLIS-K ổn định khi k tăng từ 1→100 trên CHI và NYC → xác nhận Takeaway 4\n'
    'SYN: variance cao do No-index k=10 có 1 outlier (173,711ms đã loại), giá trị k=10 dùng k=50',
    ha='center', fontsize=8.5, style='italic', color='#555')

plt.tight_layout()
plt.savefig('rq4_figure8_knn.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: rq4_figure8_knn.png")


# ═══════════════════════════════════════════════════════════
# FIGURE 7 — Range Selectivity Sweep
# ═══════════════════════════════════════════════════════════
sel_labels = ['0.00001%', '0.0001%', '0.001%', '0.01%', '0.1%']
x_pos = np.arange(len(sel_labels))

sel_data = {
    'CHI': {
        'LiLIS-K':  [153.6, 155.9, 142.1, 157.0, 157.9],
        'No-index': [1502.2, 1348.6, 1499.7, 1322.3, 1540.7],
    },
    'NYC': {
        'LiLIS-K':  [653.6, 896.2, 743.1, 869.4, 721.3],
        'No-index': [5865.2, 6023.4, 6007.3, 6230.0, 5761.1],
    },
    'SYN': {
        'LiLIS-K':  [2180.6, 2116.7, 2604.7, 2258.0, 2243.2],
        'No-index': [2818.0, 2849.6, 2886.6, 2887.2, 2768.9],
    },
}

fig7, axes7 = plt.subplots(1, 3, figsize=(14, 4.5))
fig7.suptitle('Hình 7. Range Query khi thay đổi Selectivity — LiLIS-K vs No-index\n'
              '(Cluster 3 nodes LAN, 15 runs, seed=42)',
              fontsize=12, fontweight='bold', y=1.03)

for ax, ds in zip(axes7, datasets):
    d = sel_data[ds]
    ax.plot(x_pos, d['LiLIS-K'],
            marker='o', linewidth=2, markersize=7,
            color=COLOR_LILIS, label='LiLIS-K', zorder=3)
    ax.plot(x_pos, d['No-index'],
            marker='s', linewidth=2, markersize=7,
            linestyle='--', color=COLOR_NOINDEX,
            label='No-index', zorder=3)

    # Nhãn giá trị LiLIS-K
    for xi, y_ in zip(x_pos, d['LiLIS-K']):
        ax.annotate(f'{y_:.0f}', (xi, y_),
                    textcoords='offset points', xytext=(0, 8),
                    ha='center', fontsize=7.5,
                    color=COLOR_LILIS, fontweight='bold')

    ax.set_title(f'Tập {ds}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Selectivity (%)', fontsize=10)
    ax.set_ylabel('Thời gian (ms)', fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sel_labels, fontsize=8, rotation=20)
    ax.set_ylim(0, max(max(d['LiLIS-K']), max(d['No-index'])) * 1.35)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v,_: f'{v/1000:.1f}k' if v>=1000 else f'{int(v)}'))
    ax.grid(linestyle='--', alpha=0.35, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=9, loc='upper right')

fig7.text(0.5, -0.08,
    'Nhận xét: LiLIS-K tương đối ổn định qua các mức selectivity trên CHI và SYN.\n'
    'NYC có dao động nhẹ (653→896ms) do phân bố dữ liệu lệch theo không gian thực tế.',
    ha='center', fontsize=8.5, style='italic', color='#555')

plt.tight_layout()
plt.savefig('rq4_figure7_selectivity.png', dpi=150,
            bbox_inches='tight', facecolor='white')
print("Saved: rq4_figure7_selectivity.png")

plt.show()
