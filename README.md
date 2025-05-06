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

## 随机森林模型优化指南

本文档提供了关于如何训练和复现优化后的随机森林模型的详细说明。

### 环境准备

1. 确保已安装Python环境（推荐Python 3.9+）
2. 安装所需依赖：
   ```bash
   pip install -r requirements.txt
   ```

### 数据准备

项目使用PHM 2010数据集，请确保数据已放置在指定位置：
```
data/raw/PHM_2010/
```

### 训练优化后的随机森林模型

执行以下命令来训练优化版本的随机森林模型：

```bash
python scripts/train_enhanced_rf.py --config config/rf_optimized.yaml
```

这个命令将使用`config/rf_optimized.yaml`中的优化配置训练随机森林模型，并将结果保存在`results/`目录下。

### 关于优化配置

当前最优配置有以下特点：

1. **特征工程**：
   - 使用小波包变换（3级分解）
   - 提取时域与频域特征
   - 添加统计特征和交互特征

2. **特征选择**：
   - 使用相关性筛选（Spearman相关系数≥0.75，Pearson相关系数≥0.8）
   - 递归特征消除
   - 特征重要性阈值：0.03

3. **随机森林参数**：
   - 树数量：700
   - 最大深度：8
   - 最小分裂样本数：10
   - 叶节点最小样本数：6
   - 特征采样比例：0.6
   - 样本采样比例：0.7
   - 成本复杂度剪枝系数：0.02

4. **集成学习**：
   - 使用Stacking方法
   - 基模型包含3个不同配置的随机森林
   - 元模型使用Ridge回归

### 模型性能

优化后的随机森林模型在测试集（c6实验数据）上的性能：
- MAE：17.59
- RMSE：21.55
- R²：0.711

### 结果说明

训练后，结果将保存在`results/`目录下，包含：
- 训练好的模型
- 评估结果
- 特征重要性可视化
- 预测结果

请注意：`results/`目录已被添加到`.gitignore`文件中，不会被上传到GitHub。

### 进一步优化方向

如需进一步提升性能，可考虑：
1. 收集更多训练数据
2. 尝试其他类型的模型（如梯度提升树）
3. 进一步细化特征工程
4. 针对不同工况单独训练模型

如有任何问题，请参考项目文档或联系项目维护者。
