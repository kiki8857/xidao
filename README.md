# 锡刀切削刀具磨损预测系统

基于PHM2010数据集的切削刀具磨损预测系统，使用机器学习和深度学习方法预测刀具磨损。

## 项目特点

- 数据预处理与特征工程
- 基于BP神经网络的刀具磨损预测
- 使用贝叶斯优化自动调整超参数
- 可视化分析预测结果

## 主要功能

- 原始切削力信号的预处理与清洗
- 时域、频域和小波域特征提取
- 特征选择与降维
- 基于随机森林的特征重要性分析
- 基于BP神经网络的磨损预测
- 超参数调优与模型评估
- 结果可视化

## 项目结构

```
xidao/
├── config/           # 配置文件
├── data/             # 数据文件夹
│   ├── raw/          # 原始数据
│   ├── processed/    # 处理后的数据
│   └── examples/     # 示例数据
├── scripts/          # 脚本文件
├── results/          # 结果保存
└── logs/             # 日志文件
```

## 数据说明

本项目使用PHM2010数据集进行刀具磨损预测。由于原始数据集较大（约17GB），仓库中只包含少量示例数据。

### 示例数据

`data/examples/` 目录包含少量示例数据文件，用于测试脚本功能。

### 获取完整数据集

完整的PHM2010数据集可以从以下来源获取：

1. PHM 2010官方网站：https://www.phmsociety.org/competition/phm/10
2. NASA数据仓库：https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

下载完整数据集后，将文件解压到 `data/raw/PHM_2010/` 目录。

## 使用方法

1. 安装依赖
```
pip install -r requirements.txt
```

2. 预处理数据
```
python scripts/preprocess_data.py --config config/config.yaml
```

3. 提取特征
```
python scripts/engineer_features.py --config config/config.yaml
```

4. 训练模型
```
python scripts/train_evaluate_bpnn.py --config config/config.yaml
```

5. 调整超参数
```
python scripts/tune_hyperparameters.py --config config/config.yaml --model bpnn
```

## 技术栈

- Python 3.9+
- PyTorch
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- Optuna (贝叶斯优化)
- PyWavelets (小波分析)

## 结果复现指南

### 1. 删除已有的results目录

在运行实验前，可以选择删除已有的results目录，以保持工作空间整洁。可以使用以下命令：

```bash
# 删除全部results目录（确保您已经备份了任何需要保留的重要结果）
rm -rf results/

# 或选择性删除特定结果目录
rm -rf results/rf_enhanced_*
```

### 2. 运行最优随机森林模型

我们的最优模型配置已保存在`config/rf_optimized.yaml`文件中。该配置采用平衡泛化策略，在测试集上实现了较好的性能指标（R²: 0.711）。

按照以下步骤复现最优结果：

1. 确保环境已正确配置
   ```bash
   # 激活虚拟环境（如果使用）
   conda activate xidao
   ```

2. 运行优化的随机森林训练脚本
   ```bash
   python scripts/train_enhanced_rf.py --config config/rf_optimized.yaml
   ```

3. 训练完成后，结果将保存在`results/rf_enhanced_[timestamp]/`目录中，包括：
   - 训练日志 (`rf_enhanced.log`)
   - 模型文件 (`rf_ensemble_model.joblib`)
   - 特征重要性分析 (`feature_importance.csv`和`final_feature_importance.csv`)
   - 测试集预测结果 (`ensemble_test_predictions.csv`)
   - 评估指标 (`ensemble_evaluation_results.yaml`)
   - 可视化结果 (`visualizations/`目录)

### 性能指标

当前最优配置在测试集上的性能：
- MAE: 17.59
- RMSE: 21.55
- R²: 0.711

### 模型配置特点

当前最优配置（平衡泛化版本）的主要特点：

1. **特征工程**：
   - 使用小波包变换（层级：3）
   - 保留统计特征和交互特征
   - 特征选择采用中等强度阈值（重要性阈值：0.03）

2. **集成模型**：
   - 采用stacking集成策略
   - 基础模型包含3个变体配置的随机森林
   - 元模型使用Ridge回归

3. **正则化策略**：
   - 适中的树深度控制（max_depth: 8）
   - 平衡的树数量（n_estimators: 700）
   - 合理的剪枝参数（ccp_alpha: 0.02）

### 进一步改进方向

要进一步提升模型性能，可以考虑：

1. 收集更多的训练数据
2. 探索其它类型模型（如梯度提升树、深度神经网络）
3. 进行更细致的特征工程
4. 尝试不同的数据预处理策略

注意：根据`.gitignore`配置，`results/`目录不会被提交到GitHub仓库，确保您的实验结果不会占用远程仓库空间。
