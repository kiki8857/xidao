# 数据驱动的铣刀寿命预测

本项目旨在根据 `开题.md` 文件中的研究内容，利用多源传感信号（如振动、切削力、声发射）和机器学习/深度学习技术，预测铣刀的剩余使用寿命（RUL）。

## 项目结构

```
xidao/
├── config/                  # 配置文件目录
│   └── config.yaml          # 项目配置 (数据路径, 模型参数等)
├── data/
│   ├── raw/                 # 原始数据目录 (例如 PHM2010 数据集)
│   └── processed/           # 处理后的数据 (特征文件等)
├── notebooks/               # Jupyter Notebooks (探索性分析, 实验)
├── src/
│   ├── data_processing/     # 数据处理模块
│   │   ├── __init__.py
│   │   ├── loader.py          # 数据加载
│   │   ├── preprocessing.py   # 信号预处理 (去噪, 平滑)
│   │   ├── feature_extraction.py # 特征提取 (时域, 频域, 时频域)
│   │   ├── feature_selection.py  # 特征筛选 (相关性分析)
│   │   └── dimensionality_reduction.py # 特征降维 (PCA)
│   ├── models/              # 模型定义模块
│   │   ├── __init__.py
│   │   ├── bpnn.py            # BP 神经网络模型
│   │   ├── random_forest.py   # 随机森林模型
│   │   └── model_fusion.py    # 模型融合策略
│   ├── training/            # 模型训练与评估模块
│   │   ├── __init__.py
│   │   ├── train.py           # 主训练脚本
│   │   ├── evaluate.py        # 模型评估脚本
│   │   └── tuning.py          # 超参数调优 (贝叶斯优化)
│   └── utils/               # 工具函数模块
│       ├── __init__.py
│       ├── logger.py          # 日志记录
│       └── plotting.py        # 结果可视化
├── requirements.txt         # 项目依赖库
└── README.md                # 项目说明文档
```

## 主要研究内容

1.  **信号采集与特征提取**: 处理多源信号，提取时域、频域、时频域特征。
2.  **特征筛选与降维**: 使用相关性分析筛选特征，PCA 降维。
3.  **模型构建与调优**: 实现并对比随机森林 (RF) 和反向传播神经网络 (BPNN)，使用贝叶斯优化进行超参数调整。
4.  **模型评估**: 使用 MAE, RMSE, R² 等指标评估模型性能。
5.  **(可选) 模型融合**: 探索 RF 和 BPNN 的融合策略。

## 如何运行

1.  **安装依赖**: `pip install -r requirements.txt`
2.  **准备数据**: 将原始数据放入 `data/raw/` 目录。
3.  **修改配置**: 根据需要调整 `config/config.yaml` 中的参数。
4.  **运行训练**: `python src/training/train.py` (或通过其他入口脚本) # xidao
