import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

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
# 使用 scipy.stats.linregress 可以同时获得 R^2
slope, intercept, r_value, p_value, std_err = linregress(df['Rf'], df['log_mw'])
r_squared = r_value**2

# --- 4. 绘图 ---
# 设置Seaborn的绘图风格，移除网格线
sns.set_theme(style="white") # 将风格设置为 "white" 以移除默认网格线

# 创建一个图形和坐标轴
plt.figure(figsize=(8, 6))
ax = plt.gca() # 获取当前轴

# 绘制散点 (用于图例)
# 我们先用 ax.scatter 绘制散点，并为其添加图例标签
ax.scatter(
    df['Rf'],
    df['log_mw'],
    color='#8D2F25', # 使用你提供的红棕色
    s=80, alpha=0.9, edgecolor='w', markeredgewidth=1.5, # 散点样式
    label="Data Points" # 为散点添加图例标签
)

# 使用 seaborn.regplot 绘制回归线和95%置信区间
# 注意：这里设置 scatter=False，因为我们已经用 ax.scatter 绘制了散点
sns.regplot(
    x='Rf',
    y='log_mw',
    data=df,
    color='#8D2F25',  # 使用你提供的红棕色
    scatter=False, # 不让regplot绘制散点，只绘制线和置信区间
    line_kws={'linewidth': 2.5}, # 自定义回归线样式
    ax=ax,
    label="Linear Fit" # 为回归线添加图例标签
)

# --- 5. 在图上添加方程和 R^2 值 (加粗) ---
# 格式化方程文本
# 注意：斜率是负数，直接用 + 会出现 "+ -"，所以我们用 f-string 的格式化来处理
eq_text = f"$log(Mr) = {intercept:.3f} {slope:+.3f} \cdot Rf$"
r2_text = f"$R^2 = {r_squared:.4f}$"

# 将文本放置在图的右上角，并加粗
ax.text(0.95, 0.95, eq_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='right',
        fontweight='bold', # 加粗
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.6, ec='none'))

ax.text(0.95, 0.87, r2_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='right',
        fontweight='bold', # 加粗
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.6, ec='none'))


# --- 6. 美化图表 ---
# 设置标题和坐标轴标签
ax.set_title("SDS-PAGE Standard Curve", fontsize=16, fontweight='bold')
ax.set_xlabel("Relative Mobility ($R_f$)", fontsize=14)
ax.set_ylabel("Logarithm of Molecular Weight ($log(Mr)$)", fontsize=14)

# 调整坐标轴范围，留出一些空白
ax.set_xlim(df['Rf'].min() - 0.05, df['Rf'].max() + 0.05)
ax.set_ylim(df['log_mw'].min() - 0.1, df['log_mw'].max() + 0.1)

# 加粗左边和底部的坐标轴线，并隐藏顶部和右侧轴线
ax.spines['bottom'].set_linewidth(2) # 加粗底部轴线
ax.spines['left'].set_linewidth(2)   # 加粗左侧轴线
ax.spines['top'].set_visible(False)    # 隐藏顶部轴线
ax.spines['right'].set_visible(False)  # 隐藏右侧轴线

# 添加图例
ax.legend(loc='upper left', frameon=False, fontsize=12) # 效果图的legend没有边框

# 调整布局，防止标签被裁剪
plt.tight_layout()

# --- 7. 保存并显示图片 ---
# 推荐保存为文件，以获得高分辨率图像
plt.savefig('SDS_PAGE_Standard_Curve_Final.png', dpi=300)

# 显示图形
plt.show()

# --- 8. (可选) 打印计算出的样品分子量 ---
sample_dist = 2.8
Rf_sample = sample_dist / dye_dist
log_mw_sample = intercept + slope * Rf_sample
mw_sample = 10 ** log_mw_sample
print(f"拟合方程: log(Mr) = {intercept:.3f} + {slope:.3f} * Rf")
print(f"R-squared: {r_squared:.4f}")
print("-" * 30)
print(f"样品 Rf = {Rf_sample:.3f}")
print(f"样品 Mr ≈ {mw_sample:.2f} kD")
