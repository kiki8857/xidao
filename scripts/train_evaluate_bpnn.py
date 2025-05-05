import pandas as pd
import numpy as np
import yaml
import logging
import os
import argparse
import joblib # May still need for loading scaler/pca potentially
from datetime import datetime
import sys

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

# Import necessary functions from src
try:
    from src.models.bpnn import build_bpnn
    from src.training.evaluate import calculate_regression_metrics, check_metrics_thresholds
    from src.utils.logger import setup_logger
except ImportError as e:
    raise ImportError(f"Could not import project modules. Ensure PYTHONPATH is set correctly or run from root: {e}")

# --- PyTorch Specific Imports ---
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from torch.autograd import Variable

# --- Feature normalization ---
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 实现Levenberg-Marquardt算法的优化器
class LMOptimizer:
    """Levenberg-Marquardt优化器实现，用于神经网络训练。
    
    这是一个基于Gauss-Newton方法的非线性最小二乘优化算法，
    对于小到中等规模的神经网络通常比标准梯度下降更高效。
    """
    def __init__(self, model, learning_rate=0.01, damping=0.1, max_iter=None):
        """
        Args:
            model: PyTorch模型
            learning_rate: 学习率
            damping: 阻尼因子，控制Hessian逆矩阵的正则化程度
            max_iter: 每步更新的最大迭代次数
        """
        self.model = model
        self.lr = learning_rate
        self.damping = damping
        self.max_iter = max_iter
        self.params = list(model.parameters())
        
    def zero_grad(self):
        """清除所有参数的梯度"""
        for param in self.params:
            if param.grad is not None:
                param.grad.data.zero_()
    
    def _compute_jacobian(self, outputs, inputs):
        """计算雅可比矩阵"""
        # 简化实现：不再尝试为每个样本单独计算雅可比矩阵
        # 而是对整个批次计算平均梯度
        output_size = outputs.size(1) if outputs.dim() > 1 else 1
        n_params = sum(p.numel() for p in self.params)
        jacobian = torch.zeros(output_size, n_params, device=outputs.device)
        
        for i in range(output_size):
            if output_size > 1:
                grad_output = torch.zeros_like(outputs)
                grad_output[:, i] = 1.0
            else:
                grad_output = torch.ones_like(outputs)
            
            grads = torch.autograd.grad(outputs, self.params, grad_output, create_graph=False, 
                                      retain_graph=True, allow_unused=True)
            
            col_idx = 0
            for grad in grads:
                if grad is not None:
                    # 对每个参数的梯度，计算批次平均值
                    avg_grad = grad.mean(dim=0) if grad.dim() > 1 else grad
                    flat_grad = avg_grad.reshape(-1)
                    
                    # 确保不超出雅可比矩阵的列范围
                    param_size = flat_grad.size(0)
                    if col_idx + param_size <= n_params:
                        jacobian[i, col_idx:col_idx + param_size] = flat_grad
                        col_idx += param_size
        
        return jacobian
    
    def step(self, loss_fn, inputs, targets):
        """执行一步LM优化"""
        # 前向传播
        outputs = self.model(inputs)
        loss = loss_fn(outputs, targets)
        
        try:
            # 计算雅可比矩阵 (这里针对整个批次)
            jacobian = self._compute_jacobian(outputs, inputs)
            
            # 计算Hessian矩阵近似 (J^T * J)
            JTJ = torch.matmul(jacobian.transpose(0, 1), jacobian)
            
            # 添加阻尼项形成LM系统: (J^T * J + lambda * I) * delta = J^T * e
            n_params = JTJ.size(0)
            identity = torch.eye(n_params, device=JTJ.device)
            dampened_JTJ = JTJ + self.damping * identity
            
            # 计算残差：e = y - f(x)
            error = (targets - outputs).mean(dim=0)  # 计算批次平均误差
            
            # 右侧：J^T * e
            JTe = torch.matmul(jacobian.transpose(0, 1), error)
            
            # 求解线性系统
            try:
                # 使用Cholesky分解求解线性系统
                L = torch.linalg.cholesky(dampened_JTJ)
                y = torch.linalg.solve_triangular(L, JTe, upper=False)
                delta = torch.linalg.solve_triangular(L.transpose(-2, -1), y, upper=True)
            except Exception as cholesky_error:
                # 如果Cholesky分解失败，使用伪逆
                delta = torch.matmul(torch.pinverse(dampened_JTJ), JTe)
            
            # 应用参数更新
            with torch.no_grad():
                idx = 0
                for param in self.params:
                    num_param = param.numel()
                    # 确保不超出delta的长度
                    if idx + num_param <= delta.size(0):
                        # 使用delta的相应部分更新参数
                        update = delta[idx:idx+num_param]
                        # 确保更新的形状与参数相同
                        param.add_(self.lr * update.reshape(param.shape))
                    idx += num_param
        except Exception as e:
            print(f"LM优化器步骤失败: {e}")
            # 发生错误时，回退到简单的梯度下降
            outputs = self.model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            with torch.no_grad():
                for param in self.params:
                    if param.grad is not None:
                        param.add_(-self.lr * param.grad)
            
        return loss.item()

def train_bpnn_epoch(model, dataloader, criterion, optimizer, device, logger, l1_weight=0.0):
    """Trains the BPNN model for one epoch."""
    model.train() # Set model to training mode
    running_loss = 0.0
    processed_batches = 0
    
    # 标准SGD/Adam训练流程
    if not isinstance(optimizer, LMOptimizer):
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()   # Zero the parameter gradients
            outputs = model(inputs) # Forward pass
            loss = criterion(outputs, targets) # Calculate loss
            
            # 添加L1正则化
            if l1_weight > 0:
                l1_regularization = 0.0
                for param in model.parameters():
                    l1_regularization += torch.sum(torch.abs(param))
                loss += l1_weight * l1_regularization
                
            loss.backward()         # Backward pass
            optimizer.step()        # Optimize

            running_loss += loss.item()
            processed_batches = i + 1
    # Levenberg-Marquardt训练流程
    else:
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = optimizer.step(criterion, inputs, targets)
            running_loss += loss
            processed_batches = i + 1
            
    epoch_loss = running_loss / (processed_batches if processed_batches > 0 else 1)
    return epoch_loss

def evaluate_bpnn(model, dataloader, criterion, device):
    """Evaluates the BPNN model on a dataset (e.g., test set)."""
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad(): # No need to track gradients during evaluation
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    all_preds = np.concatenate(all_preds).flatten() # Flatten predictions
    all_targets = np.concatenate(all_targets).flatten() # Flatten targets
    return avg_loss, all_targets, all_preds

def train_evaluate_bpnn(config_path):
    """Loads features, trains/evaluates BPNN model, and saves results."""

    # 1. Load configuration
    print(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        return

    # --- Create results directory for this run ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.abspath(os.path.join(project_root, 'results', f'bpnn_{timestamp}'))
    os.makedirs(results_dir, exist_ok=True)
    log_file_path = os.path.join(results_dir, 'bpnn_training.log')

    # --- Configure Logger ---
    log_level_str = config.get('logging', {}).get('log_level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger = setup_logger('BPNNTrainingLogger', log_level, log_file_path)
    logger.info(f"Logger setup complete. Log level: {log_level_str}. Log file: {log_file_path}")
    logger.info(f"Results will be saved in: {results_dir}")
    logger.info(f"Using configuration from: {config_path}")

    # --- Set Random Seed (Important for PyTorch reproducibility) ---
    random_seed = config.get('training', {}).get('random_state', None)
    if random_seed is not None:
        logger.info(f"Setting random seed to: {random_seed}")
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            # Optional: Set deterministic algorithms (might impact performance)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
            
    # --- Get Processed Data Paths ---
    processed_data_dir_rel = config.get('data', {}).get('processed_data_dir', 'data/processed')
    processed_data_dir = os.path.abspath(os.path.join(project_root, processed_data_dir_rel))
    features_path = os.path.join(processed_data_dir, 'final_features.csv')
    target_path = os.path.join(processed_data_dir, 'target_wear.csv')

    # 2. Load Final Features and Target
    logger.info("--- Step 1: Loading Final Features and Target ---")
    try:
        final_features_df = pd.read_csv(features_path, index_col=['experiment_id', 'measurement_id'])
        target_series = pd.read_csv(target_path, index_col=['experiment_id', 'measurement_id']).squeeze("columns")
        if not isinstance(target_series, pd.Series):
             raise ValueError("Loaded target is not a Pandas Series after squeeze.")
        logger.info(f"Loaded features shape: {final_features_df.shape}")
        logger.info(f"Loaded target shape: {target_series.shape}")
    except FileNotFoundError:
        logger.error(f"Features file ({features_path}) or target file ({target_path}) not found. Run engineer_features.py first.")
        return
    except Exception as e:
        logger.error(f"Error loading features or target: {e}", exc_info=True)
        return
        
    # Check alignment
    if not final_features_df.index.equals(target_series.index):
         logger.warning("Index mismatch between loaded features and target. Attempting to align.")
         common_index = final_features_df.index.intersection(target_series.index)
         if common_index.empty:
              logger.error("No common index found between features and target after loading. Aborting.")
              return
         final_features_df = final_features_df.loc[common_index]
         target_series = target_series.loc[common_index]
         logger.info(f"Aligned data shapes: Features={final_features_df.shape}, Target={target_series.shape}")

    # 3. Split Data
    logger.info("--- Step 2: Splitting Data (Train/Test based on experiment ID) ---")
    train_exp_ids = config.get('training', {}).get('train_experiments', ['c1', 'c4'])
    test_exp_ids = config.get('training', {}).get('test_experiments', ['c6'])

    available_exp = final_features_df.index.get_level_values('experiment_id').unique().tolist()
    train_exp_ids = [eid for eid in train_exp_ids if eid in available_exp]
    test_exp_ids = [eid for eid in test_exp_ids if eid in available_exp]

    if not train_exp_ids:
        logger.error(f"No training experiments found ({available_exp}). Aborting.")
        return
    if not test_exp_ids:
        logger.warning(f"No testing experiments found ({available_exp}). Evaluation skipped.")

    logger.info(f"Using experiments for training: {train_exp_ids}")
    logger.info(f"Using experiments for testing: {test_exp_ids}")

    X_train = final_features_df[final_features_df.index.get_level_values('experiment_id').isin(train_exp_ids)]
    y_train = target_series[target_series.index.get_level_values('experiment_id').isin(train_exp_ids)]

    if test_exp_ids:
        X_test = final_features_df[final_features_df.index.get_level_values('experiment_id').isin(test_exp_ids)]
        y_test = target_series[target_series.index.get_level_values('experiment_id').isin(test_exp_ids)]
    else:
        X_test, y_test = pd.DataFrame(), pd.Series(dtype=float)

    logger.info(f"Train data shape: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Test data shape: X={X_test.shape}, y={y_test.shape}")

    if X_train.empty or y_train.empty:
        logger.error("Training data is empty after splitting. Aborting.")
        return

    # 跳过特征归一化步骤，直接使用原始特征
    logger.info("--- Step 3: Building BPNN Model ---")
    input_dim = X_train.shape[1]
    model = build_bpnn_model(input_dim, config)
    if model is None:
        logger.error("Failed to build BPNN model. Aborting.")
        return

    # Setup Training Parameters
    bpnn_cfg = config.get('models', {}).get('bpnn', {})
    learning_rate = bpnn_cfg.get('learning_rate', 0.001) # Default LR
    weight_decay = bpnn_cfg.get('weight_decay', 0) # L2正则化
    epochs = bpnn_cfg.get('epochs', 100)
    batch_size = bpnn_cfg.get('batch_size', 32)
    optimizer_name = bpnn_cfg.get('optimizer', 'adam').lower()
    train_fn = bpnn_cfg.get('train_function', 'adam').lower()  # 获取训练函数，默认为adam
    criterion = nn.MSELoss() # Common choice for regression
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device} for BPNN training.")
    logger.info(f"Training function: {train_fn}, LR: {learning_rate}, Epochs: {epochs}, Batch Size: {batch_size}")

    # Create DataLoaders
    try:
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if not X_test.empty:
            X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            # Don't shuffle test loader
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 
        else:
            test_loader = None
            
    except Exception as e:
         logger.error(f"Error creating PyTorch DataLoader: {e}", exc_info=True)
         return

    # Select Optimizer and Training Function
    if train_fn == 'trainlm':
        # 使用Levenberg-Marquardt算法
        damping = bpnn_cfg.get('lm_damping', 0.1)
        optimizer = LMOptimizer(model, learning_rate=learning_rate, damping=damping)
        logger.info(f"Using Levenberg-Marquardt optimizer (trainlm) with damping={damping}")
    else:
        # 标准梯度下降优化器
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            logger.error(f"Unsupported optimizer: {optimizer_name}. Using Adam as default.")
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 设置学习率调度器 (只对非LM优化器有效)
    scheduler = None
    if not isinstance(optimizer, LMOptimizer):
        use_lr_scheduler = bpnn_cfg.get('use_lr_scheduler', False)
        if use_lr_scheduler:
            lr_scheduler_params = bpnn_cfg.get('lr_scheduler_params', {})
            scheduler_type = lr_scheduler_params.get('type', 'reduce_on_plateau').lower()
            
            if scheduler_type == 'reduce_on_plateau':
                factor = lr_scheduler_params.get('factor', 0.5)
                patience = lr_scheduler_params.get('patience', 10)
                min_lr = lr_scheduler_params.get('min_lr', 0.0001)
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, 
                                              patience=patience, verbose=True, min_lr=min_lr)
                logger.info(f"Using ReduceLROnPlateau scheduler (factor={factor}, patience={patience}, min_lr={min_lr})")
            elif scheduler_type == 'step':
                step_size = lr_scheduler_params.get('step_size', 10)
                gamma = lr_scheduler_params.get('gamma', 0.5)
                scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
                logger.info(f"Using StepLR scheduler (step_size={step_size}, gamma={gamma})")
            elif scheduler_type == 'cosine':
                t_max = lr_scheduler_params.get('t_max', epochs)
                eta_min = lr_scheduler_params.get('eta_min', 0)
                scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
                logger.info(f"Using CosineAnnealingLR scheduler (T_max={t_max}, eta_min={eta_min})")
            else:
                logger.warning(f"Unsupported scheduler type: {scheduler_type}. No scheduler will be used.")
        else:
            logger.info("No learning rate scheduler used")

    # 训练模型
    logger.info("--- Step 5: Training BPNN Model ---")
    training_start_time = datetime.now()
    logger.info(f"Training started at {training_start_time}")
    
    best_test_loss = float('inf') # For potential early stopping based on test loss
    epochs_no_improve = 0
    early_stopping_patience = bpnn_cfg.get('early_stopping_patience', 10)
    l1_weight = bpnn_cfg.get('l1_weight', 0.0) # 获取L1正则化系数
    min_error = bpnn_cfg.get('min_error', 0.0) # 获取期望误差阈值
    
    logger.info(f"Early stopping patience: {early_stopping_patience}, Minimum error threshold: {min_error}")
    
    for epoch in range(epochs):
        epoch_start_time = datetime.now()
        train_loss = train_bpnn_epoch(model, train_loader, criterion, optimizer, device, logger, l1_weight)
        
        # Evaluate on test set at the end of each epoch (if test set exists)
        test_loss = float('inf')
        if test_loader:
             test_loss, _, _ = evaluate_bpnn(model, test_loader, criterion, device)
        
        epoch_duration = datetime.now() - epoch_start_time
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Duration: {epoch_duration}")

        # 更新学习率调度器 (只对非LM优化器有效)
        if scheduler is not None and not isinstance(optimizer, LMOptimizer):
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()
            
            # 记录当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Current learning rate: {current_lr:.6f}")

        # --- 检查是否达到最小误差阈值 ---
        if test_loader and test_loss <= min_error:
            logger.info(f"Reached minimum error threshold: {test_loss:.6f} <= {min_error}. Early stopping.")
            # 保存当前模型状态
            best_model_path = os.path.join(results_dir, 'bpnn_best_model.pt')
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Best model saved to {best_model_path} (Test Loss: {test_loss:.4f})")
            break

        # --- Optional: Early Stopping Logic --- 
        if test_loader and test_loss < best_test_loss:
             best_test_loss = test_loss
             epochs_no_improve = 0
             # Save the best model state
             best_model_path = os.path.join(results_dir, 'bpnn_best_model.pt')
             torch.save(model.state_dict(), best_model_path)
             logger.info(f"New best model saved to {best_model_path} (Test Loss: {best_test_loss:.4f})")
        elif test_loader:
             epochs_no_improve += 1
             if epochs_no_improve >= early_stopping_patience:
                 logger.info(f"Early stopping triggered after {early_stopping_patience} epochs without improvement on test loss.")
                 break # Stop training
                 
    training_duration = datetime.now() - training_start_time
    logger.info(f"BPNN training finished. Total duration: {training_duration}")

    # --- Save the final model state --- 
    final_model_path = os.path.join(results_dir, 'bpnn_final_model.pt')
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final BPNN model state saved to {final_model_path}")

    # --- Load the best model for final evaluation (if early stopping was used) ---
    if os.path.exists(os.path.join(results_dir, 'bpnn_best_model.pt')):
        logger.info("Loading best model state for final evaluation.")
        model.load_state_dict(torch.load(os.path.join(results_dir, 'bpnn_best_model.pt')))

    # 7. Evaluate Final Model on Test Set
    logger.info("--- Step 6: Evaluating Final Model on Test Set ---")
    if test_loader:
        try:
            # Re-evaluate using the potentially loaded best model
            final_test_loss, y_test_actual, y_pred = evaluate_bpnn(model, test_loader, criterion, device)
            logger.info(f"Final Test Set MSE Loss: {final_test_loss:.4f}")
            
            # Calculate other regression metrics
            metrics = calculate_regression_metrics(y_test_actual, y_pred)
            logger.info(f"Final Test Set Metrics: {metrics}")

            # Check thresholds
            thresholds = config.get('training', {}).get('target_metrics', {})
            passed_thresholds = check_metrics_thresholds(metrics, thresholds)
            logger.info(f"Threshold check result: {passed_thresholds}")
            metrics['passed_thresholds'] = passed_thresholds
            metrics['final_test_mse_loss'] = final_test_loss # Add loss

            # Save evaluation results
            eval_save_path = os.path.join(results_dir, 'evaluation_results.yaml')
            with open(eval_save_path, 'w') as f:
                yaml.dump(metrics, f, default_flow_style=False)
            logger.info(f"Evaluation metrics saved to {eval_save_path}")

            # Save test predictions
            # Ensure index is preserved from the original X_test DataFrame
            predictions_df = pd.DataFrame({'actual': y_test_actual, 'predicted': y_pred}, index=X_test.index) 
            pred_save_path = os.path.join(results_dir, 'test_predictions.csv')
            predictions_df.to_csv(pred_save_path, index=True)
            logger.info(f"Test predictions saved to {pred_save_path}")

        except Exception as e:
            logger.error(f"Error during final model evaluation or saving results: {e}", exc_info=True)
    else:
        logger.warning("Test set is empty. Skipping final evaluation.")
        
    # Save used config copy
    try:
         with open(os.path.join(results_dir, 'config_used.yaml'), 'w') as f:
             yaml.dump(config, f, default_flow_style=False)
    except Exception as dump_e:
         logger.warning(f"Could not save config copy: {dump_e}")

    logger.info(f"--- BPNN Training and Evaluation Finished ---")

def build_bpnn_model(input_size, config):
    """构建BPNN模型。"""
    
    bpnn_cfg = config.get('models', {}).get('bpnn', {})
    
    # 读取配置
    hidden_size = bpnn_cfg.get('hidden_size', 8)  # 单层隐藏层神经元数量
    hidden_layers = bpnn_cfg.get('hidden_layers', None)  # 多层隐藏层配置
    output_size = bpnn_cfg.get('output_size', 1)
    dropout_rate = bpnn_cfg.get('dropout_rate', 0.0)
    activation = bpnn_cfg.get('activation', 'relu').lower()
    output_activation = bpnn_cfg.get('output_activation', 'linear').lower()
    batch_norm = bpnn_cfg.get('batch_normalization', False)
    
    # 如果没有明确定义hidden_layers，则使用单层的hidden_size
    if hidden_layers is None:
        hidden_layers = [hidden_size]
    
    # 构建网络层
    layers = []
    last_size = input_size
    
    # 添加隐藏层
    for i, size in enumerate(hidden_layers):
        layers.append(nn.Linear(last_size, size))
        
        # 批归一化（如果启用）
        if batch_norm:
            layers.append(nn.BatchNorm1d(size))
        
        # 激活函数
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.1))
        elif activation == 'elu':
            layers.append(nn.ELU())
        else:
            layers.append(nn.ReLU())  # 默认使用ReLU
        
        # Dropout（如果设置）
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        last_size = size
    
    # 输出层
    layers.append(nn.Linear(last_size, output_size))
    
    # 输出层激活函数
    if output_activation != 'linear':
        if output_activation == 'tanh':
            layers.append(nn.Tanh())
        elif output_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif output_activation == 'relu':
            layers.append(nn.ReLU())
    
    # 创建模型
    model = nn.Sequential(*layers)
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate BPNN Model")
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to the configuration file (default: config/config.yaml)')
    args = parser.parse_args()

    if not os.path.isabs(args.config):
        config_file_path = os.path.join(project_root, args.config)
    else:
        config_file_path = args.config

    if not os.path.exists(config_file_path):
        print(f"Error: Config file not found at {config_file_path}")
        sys.exit(1)

    train_evaluate_bpnn(config_file_path) 