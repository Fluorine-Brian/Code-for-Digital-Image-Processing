import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
# 导入用于创建自定义图例的模块
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
# <--- 关键修改：从 matplotlib.legend_handler 导入 HandlerTuple
from matplotlib.legend_handler import HandlerTuple

# --- 1. 你的实验数据 ---
marker_info = [
    (120, 1.2),  # 120 kD 条带的迁移距离
    (70,  2.0),  # 70 kD
    (55,  2.5),  # 55 kD
    (45,  3.0),  # 45 kD
    (35,  3.6),  # 35 kD
    (25,  4.2),  # 25 kD
    (15,  4.8),  # 15 kD
]
# 染料前沿的迁移距离
dye_dist = 6.0

# --- 2. 数据计算和处理 ---
marker_mw = np.array([m for m, d in marker_info], dtype=float)
marker_dist = np.array([d for m, d in marker_info], dtype=float)

# 计算相对迁移率 (Rf)
Rf = marker_dist / dye_dist
# 计算分子量的对数 (log10)
log_mw = np.log10(marker_mw)

# 将数据放入Pandas DataFrame，这是seaborn的最佳实践
df = pd.DataFrame({
    'Rf': Rf,
    'log_mw': log_mw
})

# --- 3. 进行线性回归以获取方程和 R^2 值 ---
slope, intercept, r_value, p_value, std_err = linregress(df['Rf'], df['log_mw'])
r_squared = r_value**2

# --- 4. 绘图 ---
# 设置Seaborn的绘图风格，使用 "ticks" 风格来保留刻度线
sns.set_theme(style="ticks")

# 创建一个图形和坐标轴
plt.figure(figsize=(8, 6))
ax = plt.gca() # 获取当前轴

# 定义颜色
plot_color = '#8D2F25'

# 绘制散点
ax.scatter(
    df['Rf'],
    df['log_mw'],
    color=plot_color,
    s=80, alpha=0.9, edgecolor='w',
    linewidth=1.5,
    label="Data Points" # 临时标签，后面会用自定义图例覆盖
)

# 使用 seaborn.regplot 绘制回归线和95%置信区间
sns.regplot(
    x='Rf',
    y='log_mw',
    data=df,
    color=plot_color,
    scatter=False, # 不让regplot绘制散点
    line_kws={'linewidth': 2.5},
    ax=ax,
    label="Linear Fit" # 临时标签
)

# --- 5. 美化图表 ---
# 设置标题和坐标轴标签
ax.set_title("SDS-PAGE Standard Curve", fontsize=16, fontweight='bold')
ax.set_xlabel("Relative Mobility ($R_f$)", fontsize=14)
ax.set_ylabel("Logarithm of Molecular Weight ($log(Mr)$)", fontsize=14)

# 调整坐标轴范围，留出一些空白
ax.set_xlim(df['Rf'].min() - 0.05, df['Rf'].max() + 0.05)
ax.set_ylim(df['log_mw'].min() - 0.1, df['log_mw'].max() + 0.1)

# 加粗左边和底部的坐标轴线
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
# 使用 sns.despine() 移除顶部和右侧的轴线，这是 "ticks" 风格下的标准做法
sns.despine()

# 加粗坐标轴的数字（刻度标签）
for tick in ax.get_xticklabels():
    tick.set_fontweight('bold')
for tick in ax.get_yticklabels():
    tick.set_fontweight('bold')

# 加粗坐标轴的刻度线（tick marks）
ax.tick_params(axis='both', which='both', width=2, length=6, direction='out')


# --- 6. 创建自定义图例 (右上角) ---
# 创建图例的 "handles" (即图例中的图标)
legend_handles = [
    # 数据点的图例：一个带白色边框的圆点
    Line2D([0], [0], marker='o', color='w', label='Data Points',
           markerfacecolor=plot_color, markersize=10, markeredgecolor='w'),
    # 线性拟合的图例：一个半透明的色块 + 一条实线
    (Patch(facecolor=plot_color, alpha=0.2, edgecolor=plot_color),
     Line2D([0], [0], color=plot_color, lw=2.5, label='Linear Fit'))
]
# 创建图例的标签
legend_labels = ['Data Points', 'Linear Fit']

# 显示图例
# <--- 关键修改：使用从 matplotlib.legend_handler 导入的 HandlerTuple
ax.legend(handles=legend_handles, labels=legend_labels,
          loc='upper right', frameon=False, fontsize=12,
          handler_map={tuple: HandlerTuple(ndivide=None)})


# --- 7. 在图上添加方程和 R^2 值 (加粗，位于图例正下方) ---
# 格式化方程文本
eq_text = f"$log(Mr) = {intercept:.3f} {slope:+.3f} \cdot Rf$"
r2_text = f"$R^2 = {r_squared:.4f}$"

# 调整y坐标以适应图例
text_y_start = 0.80
line_spacing = 0.07

ax.text(0.95, text_y_start, eq_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='right',
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.6, ec='none'))

ax.text(0.95, text_y_start - line_spacing, r2_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='right',
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.6, ec='none'))


# 调整布局，防止标签被裁剪
plt.tight_layout()

# --- 8. 保存并显示图片 ---
plt.savefig('SDS_PAGE_Standard_Curve_Final_V5.png', dpi=300)
plt.show()

# --- 9. (可选) 打印计算出的样品分子量 ---
sample_dist = 2.8
Rf_sample = sample_dist / dye_dist
log_mw_sample = intercept + slope * Rf_sample
mw_sample = 10 ** log_mw_sample
print(f"拟合方程: log(Mr) = {intercept:.3f} {slope:+.3f} * Rf")
print(f"R-squared: {r_squared:.4f}")
print("-" * 30)
print(f"样品 Rf = {Rf_sample:.3f}")
print(f"样品 Mr ≈ {mw_sample:.2f} kD")
