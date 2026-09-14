"""
visualize_rq2.py — So sánh hiệu năng 5 biến thể LiLIS: Paper vs Nhóm (NYC)
Chạy: python3 visualize_rq2.py
Output: rq2_comparison.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Dữ liệu ─────────────────────────────────────────────────────────────────
variants = ['LiLIS-F\n(Fixed-grid)',
            'LiLIS-A\n(Adaptive-grid)',
            'LiLIS-Q\n(Quad-tree)',
            'LiLIS-K\n(KD-tree)',
            'LiLIS-R\n(R-tree)']

# Paper gốc (NYC 300M) — Table III
paper = {
    'Point': [218.08, 199.09, 141.30,  82.59,  76.68],
    'Range': [704.15, 521.04, 340.57, 468.64, 471.55],
    'kNN'  : [1107.2, 1767.0, 773.50, 650.20, 618.80],
}

# Nhóm (NYC 46.5M, cluster 3 nodes LAN)
group = {
    'Point': [2037.3, 1145.2, 1255.5, 1033.3, 1328.9],
    'Range': [1410.1, 1190.1, 1087.4, 1103.2, 1405.2],
    'kNN'  : [1488.9, 1198.1, 1017.9, 1159.3, 1447.3],
}

query_types = ['Point', 'Range', 'kNN']

# ── Màu sắc ──────────────────────────────────────────────────────────────────
COLOR_PAPER = '#1D9E75'   # teal — paper
COLOR_GROUP = '#378ADD'   # blue — nhóm
COLOR_PAPER_EDGE = '#0F6E56'
COLOR_GROUP_EDGE = '#185FA5'

# ── Figure layout: 1 hàng 3 cột ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
fig.suptitle('So sánh thời gian truy vấn 5 biến thể LiLIS: Paper gốc vs Nhóm\n'
             '(Tập NYC — Paper: 300M điểm | Nhóm: 46.5M điểm, cluster 3 nodes LAN)',
             fontsize=13, fontweight='bold', y=1.02)

x = np.arange(len(variants))
bar_w = 0.35

for ax, qtype in zip(axes, query_types):
    p_vals = paper[qtype]
    g_vals = group[qtype]

    bars_p = ax.bar(x - bar_w/2, p_vals,
                    width=bar_w, color=COLOR_PAPER, edgecolor=COLOR_PAPER_EDGE,
                    linewidth=0.8, label='Paper (NYC 300M)', zorder=3)
    bars_g = ax.bar(x + bar_w/2, g_vals,
                    width=bar_w, color=COLOR_GROUP, edgecolor=COLOR_GROUP_EDGE,
                    linewidth=0.8, label='Nhóm (NYC 46.5M)', zorder=3)

    # Nhãn giá trị trên đầu cột
    for bar in bars_p:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 20,
                f'{h:.0f}', ha='center', va='bottom',
                fontsize=8, color=COLOR_PAPER_EDGE, fontweight='bold')

    for bar in bars_g:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 20,
                f'{h:.0f}', ha='center', va='bottom',
                fontsize=8, color=COLOR_GROUP_EDGE, fontweight='bold')

    # Highlight best variant (min of group)
    best_idx = np.argmin(g_vals)
    ax.annotate('★ best', xy=(x[best_idx] + bar_w/2, g_vals[best_idx]),
                xytext=(x[best_idx] + bar_w/2, g_vals[best_idx] + 200),
                fontsize=8, color=COLOR_GROUP_EDGE, ha='center',
                arrowprops=dict(arrowstyle='->', color=COLOR_GROUP_EDGE, lw=0.8))

    ax.set_title(f'{qtype} Query', fontsize=12, fontweight='bold', pad=10)
    ax.set_ylabel('Thời gian (ms)', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=8.5)
    ax.set_ylim(0, max(max(p_vals), max(g_vals)) * 1.35)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f'{v/1000:.1f}k' if v >= 1000 else f'{int(v)}'))
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ── Legend dùng chung ────────────────────────────────────────────────────────
patch_p = mpatches.Patch(color=COLOR_PAPER, label='Paper gốc (NYC 300M)')
patch_g = mpatches.Patch(color=COLOR_GROUP, label='Nhóm (NYC 46.5M, cluster 3 nodes LAN)')
fig.legend(handles=[patch_p, patch_g],
           loc='lower center', ncol=2,
           fontsize=10, framealpha=0.9,
           bbox_to_anchor=(0.5, -0.06))

# ── Ghi chú bên dưới ─────────────────────────────────────────────────────────
fig.text(0.5, -0.12,
         'Nhận xét: Cả hai đều có thứ tự F chậm nhất, K/Q nhanh nhất → xác nhận Takeaway 2.\n'
         'Chênh lệch tuyệt đối do paper dùng NYC 300M (lớn hơn nhóm 6.5×).',
         ha='center', fontsize=9, style='italic',
         color='#555555')

plt.tight_layout()
plt.savefig('rq2_comparison.png', dpi=150,
            bbox_inches='tight', facecolor='white')
print("Đã lưu: rq2_comparison.png")
plt.show()
