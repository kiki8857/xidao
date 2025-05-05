import pandas as pd
import numpy as np
import yaml
import logging
import os
import argparse
import joblib
from datetime import datetime
import sys # <--- Import sys

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

# 从项目中导入必要的模块
# (现在应该能直接使用绝对导入)
try:
    from src.data_processing.loader import load_phm2010_data
    from src.data_processing.preprocessing import preprocess_signals
    from src.data_processing.feature_extraction import extract_features_from_data
    from src.data_processing.feature_selection import select_features
    from src.data_processing.dimensionality_reduction import apply_pca
    from src.models.random_forest import build_random_forest
    from src.models.bpnn import build_bpnn # PyTorch BPNN
    from src.training.evaluate import calculate_regression_metrics, check_metrics_thresholds
    from src.utils.logger import setup_logger # <--- 使用绝对导入
except ImportError as e:
    # 移除之前的后备导入逻辑，因为修改 sys.path 后应该不再需要
    # print(f"Error importing project modules: {e}")
    # print("Ensure the script is run from the correct directory or PYTHONPATH is set.")
    raise ImportError(f"Could not import project modules even after adding root to sys.path: {e}")

# --- PyTorch Specific Imports (if using BPNN) ---
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# --- 移除基本日志配置 ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # 获取 logger 但不在此配置

# === 主要训练函数 ===
def train_model(config_path):
    """执行完整的模型训练流程。"""

    # 1. 加载配置
    # logger.info(f"Loading configuration from: {config_path}") # 日志在此处配置前无法使用
    print(f"Loading configuration from: {config_path}") # 使用 print 替代
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}") # 使用 print
        # logging.error(f"Error loading config file {config_path}: {e}", exc_info=True)
        return
        
    # --- 创建结果保存目录 (需要在设置 logger 前确定日志文件路径) ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type_config = config.get('training', {}).get('model_type', 'random_forest').lower()
    results_dir = os.path.abspath(os.path.join(os.path.dirname(config_path), '..', 'results', f'{model_type_config}_{timestamp}'))
    os.makedirs(results_dir, exist_ok=True)
    log_file_path = os.path.join(results_dir, 'training.log')

    # --- 配置日志记录器 --- 
    log_level_str = config.get('logging', {}).get('log_level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger = setup_logger('TrainingLogger', log_level, log_file_path)
    logger.info(f"Logger setup complete. Log level: {log_level_str}. Log file: {log_file_path}")
    logger.info(f"Results will be saved in: {results_dir}")

    # 现在可以使用 logger 了
    logger.info(f"Using configuration from: {config_path}")
    
    # 保存使用的配置文件副本
    try:
         with open(os.path.join(results_dir, 'config_used.yaml'), 'w') as f:
             yaml.dump(config, f, default_flow_style=False)
    except Exception as dump_e:
         logger.warning(f"Could not save config copy: {dump_e}")

    # --- 设置随机种子 (如果配置中指定) ---
    random_seed = config.get('training', {}).get('random_state', None)
    if random_seed is not None:
        logger.info(f"Setting random seed to: {random_seed}")
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        # Note: Setting seeds might not guarantee complete reproducibility with CUDA

    # 2. 数据加载
    logger.info("--- Step 1: Loading Data ---")
    data_cfg = config.get('data', {})
    raw_data_dir = data_cfg.get('dataset_path', 'data/raw/PHM_2010/')
    # 获取绝对路径
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '../..')) # 假设在 src/training
    if not os.path.isabs(raw_data_dir):
        raw_data_dir = os.path.abspath(os.path.join(project_root, raw_data_dir))
    
    raw_data_dict = load_phm2010_data(raw_data_dir)
    if not raw_data_dict:
        logger.error("Failed to load raw data. Aborting training.")
        return

    # 3. 数据预处理 (对每个实验分别处理)
    logger.info("--- Step 2: Preprocessing Data ---")
    processed_data = {}
    for exp_name, df in raw_data_dict.items():
        logger.info(f"Preprocessing data for experiment: {exp_name}...")
        processed_df = preprocess_signals(df, config)
        if processed_df is not None:
            processed_data[exp_name] = processed_df
        else:
            logger.warning(f"Preprocessing failed for experiment {exp_name}. Skipping.")
    if not processed_data:
        logger.error("Preprocessing failed for all experiments. Aborting.")
        return
    del raw_data_dict # 释放内存

    # 4. 特征提取
    logger.info("--- Step 3: Extracting Features ---")
    features_df, target_series = extract_features_from_data(processed_data, config)
    del processed_data # 释放内存
    if features_df is None or target_series is None:
        logger.error("Feature extraction failed. Aborting training.")
        return

    # 5. 特征选择
    logger.info("--- Step 4: Selecting Features ---")
    selected_features_df = select_features(features_df, target_series, config)
    if selected_features_df.empty or selected_features_df is None:
        logger.error("Feature selection resulted in empty or None DataFrame. Aborting training.")
        return
    # 重新对齐 target_series 到筛选后的特征索引 (因为特征选择可能移除样本?)
    target_series = target_series.loc[selected_features_df.index]
    del features_df # 释放内存

    # 6. 特征降维 (PCA)
    logger.info("--- Step 5: Applying PCA ---")
    # PCA & Scaler 需要保存，以便应用于测试集
    pca_save_path = os.path.join(results_dir, 'preprocessing_models')
    pca_result = apply_pca(selected_features_df, config, save_path=pca_save_path)
    if pca_result is None:
        logger.error("PCA failed. Aborting training.")
        return
    final_features_df, scaler, pca_model = pca_result # 假设保存成功并返回了 scaler 和 pca
    del selected_features_df # 释放内存

    # --- 数据集划分 (根据 PHM2010 规则 或 用户指定) ---
    logger.info("--- Step 6: Splitting Data (Train/Test based on experiment ID) ---")
    # 修改默认划分: c1, c4 -> 训练; c6 -> 测试
    train_exp_ids = config.get('training', {}).get('train_experiments', ['c1', 'c4']) # <-- 修改默认值
    test_exp_ids = config.get('training', {}).get('test_experiments', ['c6'])       # <-- 修改默认值
    
    # 检查数据是否存在
    available_exp = final_features_df.index.get_level_values('experiment_id').unique().tolist()
    train_exp_ids = [eid for eid in train_exp_ids if eid in available_exp]
    test_exp_ids = [eid for eid in test_exp_ids if eid in available_exp]
    
    if not train_exp_ids:
         logger.error(f"No training experiments found in the processed data ({available_exp}). Configured: {config.get('training', {}).get('train_experiments')}. Aborting.")
         return
    if not test_exp_ids:
         logger.warning(f"No testing experiments found in the processed data ({available_exp}). Configured: {config.get('training', {}).get('test_experiments')}. Evaluation will be skipped or incomplete.")
         # 可能是只加载了训练数据

    logger.info(f"Using experiments for training: {train_exp_ids}")
    logger.info(f"Using experiments for testing: {test_exp_ids}")

    X_train = final_features_df[final_features_df.index.get_level_values('experiment_id').isin(train_exp_ids)]
    y_train = target_series[target_series.index.get_level_values('experiment_id').isin(train_exp_ids)]
    
    if test_exp_ids:
        X_test = final_features_df[final_features_df.index.get_level_values('experiment_id').isin(test_exp_ids)]
        y_test = target_series[target_series.index.get_level_values('experiment_id').isin(test_exp_ids)]
    else:
        X_test, y_test = pd.DataFrame(), pd.Series(dtype=float) # Empty if no test data

    logger.info(f"Train data shape: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
    
    if X_train.empty or y_train.empty:
        logger.error("Training data is empty after splitting. Aborting.")
        return

    # 7. 模型选择与构建
    logger.info(f"--- Step 7: Building Model ({model_type_config}) ---")
    model = None
    if model_type_config == 'random_forest':
        model = build_random_forest(config, random_state=random_seed)
    elif model_type_config == 'bpnn':
        input_dim = X_train.shape[1]
        model = build_bpnn(input_dim, config)
    else:
        logger.error(f"Unsupported model_type in config: '{model_type_config}'. Choose 'random_forest' or 'bpnn'.")
        return

    if model is None:
        logger.error("Failed to build the specified model. Aborting training.")
        return

    # 8. 模型训练
    logger.info(f"--- Step 8: Training Model ({model_type_config}) ---")
    if model_type_config == 'random_forest':
        try:
            model.fit(X_train, y_train)
            logger.info("Random Forest model training complete.")
            # 保存 RF 模型
            model_save_path = os.path.join(results_dir, 'random_forest_model.joblib')
            joblib.dump(model, model_save_path)
            logger.info(f"Trained Random Forest model saved to {model_save_path}")
        except Exception as e:
            logger.error(f"Error training Random Forest: {e}", exc_info=True)
            return
            
    elif model_type_config == 'bpnn':
        # --- BPNN 训练循环 ---
        bpnn_cfg = config.get('models', {}).get('bpnn', {})
        train_cfg = config.get('training', {})
        
        learning_rate = bpnn_cfg.get('learning_rate', 0.001) # 可能从调优步骤获取?
        epochs = bpnn_cfg.get('epochs', 100)
        batch_size = bpnn_cfg.get('batch_size', 32)
        optimizer_name = bpnn_cfg.get('optimizer', 'adam').lower()
        # 损失函数 - 通常用于回归的是 MSE
        criterion = nn.MSELoss()

        # 选择设备 (CPU 或 GPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logger.info(f"Using device: {device} for BPNN training.")

        # 创建 DataLoader
        try:
            X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1) # Ensure shape [n_samples, 1]
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        except Exception as e:
             logger.error(f"Error creating PyTorch DataLoader: {e}", exc_info=True)
             return

        # 选择优化器
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) # Momentum 可配置
        else:
            logger.warning(f"Unsupported optimizer '{optimizer_name}', defaulting to Adam.")
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        logger.info(f"Starting BPNN training for {epochs} epochs...")
        model.train() # 设置为训练模式
        for epoch in range(epochs):
            epoch_loss = 0.0
            for i, (batch_X, batch_y) in enumerate(train_loader):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                # 前向传播
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            if (epoch + 1) % 10 == 0: # 每 10 个 epoch 打印一次日志
                logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.6f}')
            # TODO: 实现早停逻辑 (Early Stopping)
            # early_stopping_patience = bpnn_cfg.get('early_stopping_patience', 10)
            # 需要在验证集上评估损失并判断是否停止
        
        logger.info("BPNN training finished.")
        # 保存 BPNN 模型
        model_save_path = os.path.join(results_dir, 'bpnn_model.pth')
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"Trained BPNN model state_dict saved to {model_save_path}")

    # 9. 在测试集上评估
    logger.info(f"--- Step 9: Evaluating Model on Test Set ---")
    if X_test.empty or y_test.empty:
        logger.warning("Test set is empty. Skipping evaluation.")
    else:
        if model_type_config == 'random_forest':
            y_pred = model.predict(X_test)
        elif model_type_config == 'bpnn':
            model.eval() # 设置为评估模式
            with torch.no_grad():
                X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
                y_pred_tensor = model(X_test_tensor)
                y_pred = y_pred_tensor.cpu().numpy().squeeze() # 转换回 numpy array

        # 计算指标
        test_metrics = calculate_regression_metrics(y_test, y_pred)
        logger.info(f"Test Set Metrics: {test_metrics}")

        # 检查是否满足阈值
        target_metrics_cfg = config.get('training', {}).get('target_metrics', {})
        check_metrics_thresholds(test_metrics, target_metrics_cfg)
        
        # 保存预测结果和真实值
        try:
            results_df = pd.DataFrame({'y_true': y_test.values, 'y_pred': y_pred}, index=y_test.index)
            results_save_path = os.path.join(results_dir, 'test_predictions.csv')
            results_df.to_csv(results_save_path)
            logger.info(f"Test predictions saved to {results_save_path}")
        except Exception as save_e:
             logger.warning(f"Could not save test predictions: {save_e}")

    logger.info("--- Training and Evaluation Pipeline Finished ---")

# === 主程序入口 ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a tool wear prediction model.")
    parser.add_argument('--config', type=str, required=True, 
                        help="Path to the configuration YAML file.")
    # 可以添加其他命令行参数，例如覆盖配置中的某些项
    # parser.add_argument('--model_type', type=str, choices=['random_forest', 'bpnn'], 
    #                     help="Override the model type specified in the config.")

    args = parser.parse_args()

    # 检查配置文件路径是否存在
    if not os.path.exists(args.config):
         print(f"Error: Config file not found at {args.config}")
         exit(1)

    train_model(args.config)
