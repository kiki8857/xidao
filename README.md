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
│   └── processed/    # 处理后的数据
├── scripts/          # 脚本文件
├── results/          # 结果保存
└── logs/             # 日志文件
```

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
