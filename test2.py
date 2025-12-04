import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

# 1. 准备数据 (剔除了异常点 S7)
data = {
    # Concentration 现为横坐标 (X-axis)
    'Concentration': [1.67, 0.5, 0.17, 0.05, 0.0167, 0.005],
    # RLU 现为纵坐标 (Y-axis)
    'RLU': [165223, 82665, 29591, 8365, 2976, 878]
}

df = pd.DataFrame(data)

# 2. RLU 纵坐标单位缩放 (可选)
# 为了让公式系数更好看，这里将 RLU 除以 10000，单位变为 (x10^4 RLU)
df['RLU_Scaled'] = df['RLU'] / 10000

# 3. 计算线性回归 (y = kx + b)
# 注意：现在 x 是 Concentration，y 是 RLU_Scaled
slope, intercept, r_value, p_value, std_err = stats.linregress(df['Concentration'], df['RLU_Scaled'])
r_squared = r_value ** 2

# 4. 设置绘图风格
sns.set(style="ticks")
plt.figure(figsize=(8, 6))

# 绘制散点和拟合线 (使用 regplot 方便地绘制置信区间)
# color='#A52A2A' 是模仿参考图中的红褐色
# 使用 label 来生成图例
ax = sns.regplot(x='Concentration', y='RLU_Scaled', data=df,
                 ci=95,
                 scatter_kws={'s': 60, 'edgecolor': 'black', 'alpha': 0.8, 'label': 'Standard Points'},
                 line_kws={'color': '#A52A2A', 'alpha': 0.8, 'label': 'Linear Fit'},
                 color='#A52A2A')

# 5. 添加拟合方程和 R² 文本
# 根据斜率的正负生成公式字符串
equation_text = f'y = {slope:.4f}x + {intercept:.4f}'
if intercept < 0:
    equation_text = f'y = {slope:.4f}x - {abs(intercept):.4f}'

r2_text = f'$R^2 = {r_squared:.4f}$'

# 将文本放置在图表右上角区域 (通常标准曲线拟合文本放在图内空白处)
plt.text(x=df['Concentration'].max() * 0, y=df['RLU_Scaled'].max()*1.5,
         s=f'{equation_text}\n{r2_text}',
         fontsize=14, color='black', fontweight='bold') # 颜色改为黑色并加粗

# 6. 坐标轴标签和设置
plt.xlabel(r'Standard Concentration ($\mu g/L$)', fontsize=14, fontweight='bold')
plt.ylabel(r'RLU ($\times 10^4$)', fontsize=14, fontweight='bold')

# 坐标轴刻度值加粗
for tick in ax.get_xticklabels():
    tick.set_fontweight('bold')
for tick in ax.get_yticklabels():
    tick.set_fontweight('bold')

# 添加图例
plt.legend(loc='upper left', fontsize=12)

# 设置网格
plt.grid(True, linestyle='--', alpha=0.3)

# 移除顶部和右侧的边框
sns.despine()

# 显示图表
plt.title('Standard Curve Linear Fit', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('standard_curve_fit.png', dpi=300) # 保存为文件
plt.show()

print("\n--- 拟合结果 ---")
print(f"拟合方程 (Y为 RLU $\\times 10^4$，X为 Concentration): {equation_text}")
print(f"R方值: {r_squared:.4f}")