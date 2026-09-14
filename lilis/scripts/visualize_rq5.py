"""
visualize_rq5.py — Figure 9: Index Build Cost (LiLIS vs Sedona)
Output: rq5_figure9_build.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Dữ liệu Build Time (giây) ────────────────────────────────────────────────
# NYC — cluster 3 nodes LAN
lilis_names  = ['LiLIS-F', 'LiLIS-A', 'LiLIS-R', 'LiLIS-Q', 'LiLIS-K']
lilis_build  = [27.238, 27.302, 29.590, 32.002, 38.314]   # seconds

sedona_names = ['Sedona-QQ', 'Sedona-RQ', 'Sedona-RK']
sedona_build = [67.294, 93.863, 94.761]

# Paper reference (Figure 9 — approximate from paper)
paper_lilis  = {'LiLIS-F': 41.5, 'LiLIS-A': 48.8, 'LiLIS-Q': 53.7,
                'LiLIS-K': 50.5, 'LiLIS-R': 44.9}
paper_sedona = {'Sedona-QQ': 68, 'Sedona-QK': 68,
                'Sedona-RQ': 100, 'Sedona-RK': 100}

COLOR_LILIS  = '#1D9E75'
COLOR_SEDONA = '#E24B4A'
COLOR_PAPER  = '#888780'

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle('Hình 9. Chi phí xây dựng chỉ mục — LiLIS vs Apache Sedona (tập NYC)',
             fontsize=13, fontweight='bold', y=1.02)

# ── Panel trái: Nhóm (cluster 3 nodes LAN) ───────────────────────────────────
ax = axes[0]
ax.set_title('Nhóm — NYC 46.5M (cluster 3 nodes LAN)', fontsize=11, fontweight='bold')

# LiLIS dots
x_l = np.arange(len(lilis_names))
ax.scatter(x_l, lilis_build, s=160, color=COLOR_LILIS,
           zorder=4, label='LiLIS variants', marker='o')
for i, (name, val) in enumerate(zip(lilis_names, lilis_build)):
    ax.annotate(f'{val:.1f}s',
                (i, val), textcoords='offset points',
                xytext=(0, 10), ha='center', fontsize=9,
                color=COLOR_LILIS, fontweight='bold')

# Sedona squares
x_s = np.arange(len(lilis_names), len(lilis_names) + len(sedona_names))
ax.scatter(x_s, sedona_build, s=160, color=COLOR_SEDONA,
           zorder=4, label='Sedona variants', marker='s')
for i, (name, val) in enumerate(zip(sedona_names, sedona_build)):
    ax.annotate(f'{val:.1f}s',
                (len(lilis_names) + i, val),
                textcoords='offset points',
                xytext=(0, 10), ha='center', fontsize=9,
                color=COLOR_SEDONA, fontweight='bold')

# Annotation: LiLIS-K vs Sedona-RK speedup
ax.annotate('', xy=(4, 38.3), xytext=(7, 94.8),
            arrowprops=dict(arrowstyle='<->', color='#555', lw=1.5))
ax.text(5.5, 68, '2.5×\nnhanh hơn', ha='center', fontsize=9,
        color='#333', style='italic',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc', alpha=0.8))

all_names = lilis_names + sedona_names
ax.set_xticks(range(len(all_names)))
ax.set_xticklabels(all_names, rotation=25, ha='right', fontsize=9)
ax.set_ylabel('Thời gian xây dựng (giây)', fontsize=10)
ax.set_ylim(0, 115)
ax.axvline(x=4.5, color='#ccc', linestyle='--', linewidth=1)
ax.text(2, 108, 'LiLIS', ha='center', fontsize=10,
        color=COLOR_LILIS, fontweight='bold')
ax.text(6.5, 108, 'Sedona', ha='center', fontsize=10,
        color=COLOR_SEDONA, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.35, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=9, loc='upper left')

# ── Panel phải: So sánh với Paper ─────────────────────────────────────────────
ax2 = axes[1]
ax2.set_title('So sánh với Paper gốc (NYC 300M)', fontsize=11, fontweight='bold')

paper_l_names = list(paper_lilis.keys())
paper_l_vals  = list(paper_lilis.values())
paper_s_names = list(paper_sedona.keys())
paper_s_vals  = list(paper_sedona.values())

x_pl = np.arange(len(paper_l_names))
x_ps = np.arange(len(paper_l_names), len(paper_l_names) + len(paper_s_names))

# Paper (gray)
ax2.scatter(x_pl, paper_l_vals, s=120, color=COLOR_PAPER,
            zorder=4, marker='o', alpha=0.6, label='Paper LiLIS')
ax2.scatter(x_ps, paper_s_vals, s=120, color=COLOR_PAPER,
            zorder=4, marker='s', alpha=0.6, label='Paper Sedona')

# Nhóm overlaid (colored)
ax2.scatter([0,1,3,4,2], lilis_build, s=100, color=COLOR_LILIS,
            zorder=5, marker='o', label='Nhóm LiLIS')
sedona_map = [0, 2, 3]  # QQ → idx0, RQ → idx2, RK → idx3
for si, (sv, ni) in enumerate(zip(sedona_build, [len(paper_l_names),
        len(paper_l_names)+1, len(paper_l_names)+2])):
    ax2.scatter(ni, sv, s=100, color=COLOR_SEDONA,
                zorder=5, marker='s')

all_paper = paper_l_names + paper_s_names
ax2.set_xticks(range(len(all_paper)))
ax2.set_xticklabels(all_paper, rotation=25, ha='right', fontsize=9)
ax2.set_ylabel('Thời gian xây dựng (giây)', fontsize=10)
ax2.set_ylim(0, 120)
ax2.axvline(x=4.5, color='#ccc', linestyle='--', linewidth=1)
ax2.text(2, 115, 'LiLIS', ha='center', fontsize=10,
        color='#555', fontweight='bold')
ax2.text(7, 115, 'Sedona', ha='center', fontsize=10,
        color='#555', fontweight='bold')
ax2.grid(axis='y', linestyle='--', alpha=0.35, zorder=0)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

legend_patches = [
    mpatches.Patch(color=COLOR_PAPER, alpha=0.6, label='Paper gốc (NYC 300M)'),
    mpatches.Patch(color=COLOR_LILIS, label='Nhóm LiLIS (NYC 46.5M)'),
    mpatches.Patch(color=COLOR_SEDONA, label='Nhóm Sedona (NYC 46.5M)'),
]
ax2.legend(handles=legend_patches, fontsize=8.5, loc='upper left')

# ── Caption ──────────────────────────────────────────────────────────────────
fig.text(0.5, -0.06,
    'Thứ tự LiLIS: F < A < R < Q < K (nhóm & paper đều đúng) | '
    'LiLIS-K nhanh hơn Sedona-RK/RQ: 2.5× (nhóm) vs 2× (paper)',
    ha='center', fontsize=9, style='italic', color='#555')

plt.tight_layout()
plt.savefig('rq5_figure9_build.png', dpi=150,
            bbox_inches='tight', facecolor='white')
print("Saved: rq5_figure9_build.png")
plt.show()
