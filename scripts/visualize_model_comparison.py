import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 创建模型性能数据
models = [
    'BPNN (Initial)', 
    'BPNN (Bayesian Opt)', 
    'BPNN (3-layer WPD)',
    'Random Forest (Initial)', 
    'Random Forest (3-layer WPD)'
]

mae_values = [18.00, 15.37, 20.07, 23.63, 27.84]
rmse_values = [22.15, 17.97, 21.30, 25.74, 30.67]
r2_values = [0.69, 0.80, 0.72, 0.59, 0.41]

# 创建图形目录
os.makedirs('results/visualizations', exist_ok=True)

# 设置宽度
bar_width = 0.25
index = np.arange(len(models))

# 创建图形
plt.figure(figsize=(14, 10))

# 创建三个子图
plt.subplot(3, 1, 1)
bars1 = plt.bar(index, mae_values, bar_width, label='MAE', color='#3498db')
plt.axhline(y=5.0, color='r', linestyle='--', alpha=0.7, label='MAE Target (5.0)')
plt.ylabel('MAE (Lower is Better)')
plt.title('MAE Comparison Across Models')
plt.xticks([])
for i, v in enumerate(mae_values):
    plt.text(i, v + 0.5, f'{v:.2f}', ha='center')
plt.legend()

plt.subplot(3, 1, 2)
bars2 = plt.bar(index, rmse_values, bar_width, label='RMSE', color='#e74c3c')
plt.axhline(y=5.0, color='r', linestyle='--', alpha=0.7, label='RMSE Target (5.0)')
plt.ylabel('RMSE (Lower is Better)')
plt.title('RMSE Comparison Across Models')
plt.xticks([])
for i, v in enumerate(rmse_values):
    plt.text(i, v + 0.5, f'{v:.2f}', ha='center')
plt.legend()

plt.subplot(3, 1, 3)
bars3 = plt.bar(index, r2_values, bar_width, label='R²', color='#2ecc71')
plt.axhline(y=0.85, color='r', linestyle='--', alpha=0.7, label='R² Target (0.85)')
plt.ylabel('R² (Higher is Better)')
plt.title('R² Comparison Across Models')
plt.xticks(index, models, rotation=15, ha='right')
for i, v in enumerate(r2_values):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
plt.legend()

plt.tight_layout()
plt.savefig('results/visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
print("Model comparison visualization saved to: results/visualizations/model_comparison.png") 