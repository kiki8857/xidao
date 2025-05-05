import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 加载预测结果
pred_df = pd.read_csv('results/bpnn_20250505_184404/test_predictions.csv')

# 从评估结果中直接读取指标
r2 = 0.7990  # 从之前输出中获取
mae = 15.37  # 从之前输出中获取
rmse = 17.97  # 从之前输出中获取

# 创建图形
plt.figure(figsize=(10, 6))
plt.scatter(pred_df['actual'], pred_df['predicted'], alpha=0.6)
plt.plot([pred_df['actual'].min(), pred_df['actual'].max()], 
         [pred_df['actual'].min(), pred_df['actual'].max()], 
         'r--', linewidth=2)

# 添加标题和标签
plt.title(f'BPNN Model Prediction Results (R² = {r2:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f})')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True, linestyle='--', alpha=0.7)

# 创建结果目录如果不存在
os.makedirs('results/visualizations', exist_ok=True)

# 保存图像
plt.savefig('results/visualizations/bpnn_prediction_visualization.png', dpi=300, bbox_inches='tight')
print('Visualization saved to: results/visualizations/bpnn_prediction_visualization.png') 