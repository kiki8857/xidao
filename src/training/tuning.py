import pandas as pd
import numpy as np
import yaml
import logging
import os
import argparse
import joblib
from datetime import datetime
import sys

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

# Data Processing & Model Building Modules
try:
    from src.data_processing.loader import load_phm2010_data
    from src.data_processing.preprocessing import preprocess_signals
    from src.data_processing.feature_extraction import extract_features_from_data
    from src.data_processing.feature_selection import select_features
    from src.data_processing.dimensionality_reduction import apply_pca
    from src.models.random_forest import build_random_forest
    from src.models.bpnn import build_bpnn, BPNN # Import BPNN class for type hinting if needed
    from src.training.evaluate import calculate_regression_metrics
    from src.utils.logger import setup_logger
except ImportError as e:
    raise ImportError(f"Could not import project modules even after adding root to sys.path: {e}")

# Scikit-learn & Scikit-optimize
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor # Needed for type hinting
from sklearn.exceptions import NotFittedError
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

# PyTorch (if tuning BPNN)
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Global variables to hold data for objective function
# This avoids passing large data structures repeatedly
X_train_global = None
y_train_global = None
config_global = None
model_type_global = None
scaler_global = None
pca_model_global = None

# 初始化全局logger
logger = logging.getLogger('TuningLogger')
def initialize_logger(log_file_path, log_level=logging.INFO):
    """初始化或重置logger"""
    global logger
    logger = setup_logger('TuningLogger', log_level, log_file_path)
    return logger

# === Objective Function for Bayesian Optimization ===

def objective(**params):
    """Objective function for skopt to minimize (e.g., validation MAE)."""
    global X_train_global, y_train_global, config_global, model_type_global, scaler_global, pca_model_global

    if X_train_global is None or y_train_global is None or config_global is None:
        logger.error("Global data/config not set for objective function.")
        return np.inf # Return a high value indicating failure

    # 处理hidden_layers_str参数（如果存在）
    if 'hidden_layers_str' in params:
        try:
            # 将字符串表示转换回Python列表
            hidden_layers_str = params.pop('hidden_layers_str')
            # 安全的eval，仅允许列表、整数和逗号
            import ast
            params['hidden_layers'] = ast.literal_eval(hidden_layers_str)
            logger.debug(f"解析hidden_layers: {params['hidden_layers']}")
        except Exception as e:
            logger.error(f"解析hidden_layers_str失败: {e}, value={hidden_layers_str}")
            return np.inf

    tuning_cfg = config_global.get('tuning', {})
    training_cfg = config_global.get('training', {})
    n_splits = training_cfg.get('cv_folds', 5)
    random_seed = training_cfg.get('random_state', None)
    target_metric = tuning_cfg.get('optimization_metric', 'mae').lower()
    if target_metric not in ['mae', 'rmse']:
        logger.warning(f"Unsupported optimization_metric '{target_metric}'. Defaulting to 'mae'.")
        target_metric = 'mae'

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    scores = []

    logger.debug(f"Evaluating parameters: {params}")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_global, y_train_global)):
        X_fold_train, X_fold_val = X_train_global.iloc[train_idx], X_train_global.iloc[val_idx]
        y_fold_train, y_fold_val = y_train_global.iloc[train_idx], y_train_global.iloc[val_idx]
        
        # --- Fit Scaler and PCA ONLY on this fold's training data --- 
        # This prevents data leakage from validation fold into the preprocessor fitting
        try:
            # Scaling
            fold_scaler = scaler_global.__class__() # Create new instance
            X_fold_train_scaled = fold_scaler.fit_transform(X_fold_train)
            X_fold_val_scaled = fold_scaler.transform(X_fold_val)
            
            # PCA
            fold_pca = pca_model_global.__class__(n_components=pca_model_global.n_components) # Create new instance
            X_fold_train_pca = fold_pca.fit_transform(X_fold_train_scaled)
            X_fold_val_pca = fold_pca.transform(X_fold_val_scaled)

            X_fold_train_final = pd.DataFrame(X_fold_train_pca, index=X_fold_train.index)
            X_fold_val_final = pd.DataFrame(X_fold_val_pca, index=X_fold_val.index)
            
        except Exception as preprocess_e:
            logger.error(f"Error during preprocessing within fold {fold+1}: {preprocess_e}")
            scores.append(np.inf) # Penalize this parameter set heavily
            continue # Skip to next fold
            
        input_dim = X_fold_train_final.shape[1]

        # --- Build and Train Model --- 
        try:
            if model_type_global == 'random_forest':
                # Build RF with current hyperparams
                temp_config = config_global.copy()
                temp_config['models']['random_forest'].update(params) # Inject tuned params
                model = build_random_forest(temp_config, random_state=random_seed)
                if model is None: raise ValueError("Failed to build RF model in objective.")
                
                model.fit(X_fold_train_final, y_fold_train)
                y_pred = model.predict(X_fold_val_final)

            elif model_type_global == 'bpnn':
                # Build BPNN with current hyperparams
                temp_config = config_global.copy()
                temp_config['models']['bpnn'].update(params) # Inject tuned params 
                model = build_bpnn(input_dim, temp_config)
                if model is None: raise ValueError("Failed to build BPNN model in objective.")
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                
                bpnn_cfg = temp_config['models']['bpnn']
                learning_rate = params.get('learning_rate', bpnn_cfg.get('learning_rate', 0.001))
                weight_decay = params.get('weight_decay', bpnn_cfg.get('weight_decay', 0.0))
                epochs = min(bpnn_cfg.get('epochs', 50), 100)  # 在调优时使用较少的轮次以加快速度
                batch_size = params.get('batch_size', bpnn_cfg.get('batch_size', 32))
                # 将numpy.int64转换为Python原生int类型
                if hasattr(batch_size, 'item'):
                    batch_size = int(batch_size.item())
                else:
                    batch_size = int(batch_size)
                optimizer_name = bpnn_cfg.get('optimizer', 'adam').lower()
                criterion = nn.MSELoss()
                
                X_t = torch.tensor(X_fold_train_final.values, dtype=torch.float32)
                y_t = torch.tensor(y_fold_train.values, dtype=torch.float32).unsqueeze(1)
                fold_dataset = TensorDataset(X_t, y_t)
                fold_loader = DataLoader(fold_dataset, batch_size=batch_size, shuffle=True)
                
                if optimizer_name == 'adam': 
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                else: 
                    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

                model.train()
                for epoch in range(epochs):
                    for batch_X, batch_y in fold_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                
                # Predict on validation set
                model.eval()
                with torch.no_grad():
                    X_v_tensor = torch.tensor(X_fold_val_final.values, dtype=torch.float32).to(device)
                    y_pred_tensor = model(X_v_tensor)
                    y_pred = y_pred_tensor.cpu().numpy().squeeze()

            else:
                raise ValueError(f"Unsupported model type in objective: {model_type_global}")

            # --- Evaluate Performance on Validation Fold ---
            fold_metrics = calculate_regression_metrics(y_fold_val, y_pred)
            if not fold_metrics:
                 scores.append(np.inf) # Failed calculation
            else:
                 score = fold_metrics.get(target_metric, np.inf)
                 scores.append(score if not np.isnan(score) else np.inf)
        
        except NotFittedError as nfe:
             logger.error(f"Model not fitted error during fold {fold+1} evaluation: {nfe}. Params: {params}")
             scores.append(np.inf)
        except Exception as e:
            logger.error(f"Error during model training/evaluation in fold {fold+1}: {e}. Params: {params}", exc_info=True)
            scores.append(np.inf) # Penalize failures

    average_score = np.mean(scores)
    logger.info(f"Evaluated params: {params} -> Avg {target_metric.upper()}: {average_score:.6f}")
    return average_score

# === Main Tuning Function ===
def tune_hyperparameters(config_path):
    """执行超参数调优流程。"""
    global X_train_global, y_train_global, config_global, model_type_global, scaler_global, pca_model_global

    # 1. Load Config
    print(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config_global = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        return
        
    # Set random seed globally for reproducibility of tuning process
    random_seed = config_global.get('training', {}).get('random_state', None)
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        # skopt uses numpy random state

    model_type_global = config_global.get('training', {}).get('model_type', 'random_forest').lower()
    print(f"Starting hyperparameter tuning for model type: {model_type_global}")
    
    # --- Create results directory for tuning --- 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tuning_results_dir = os.path.abspath(os.path.join(os.path.dirname(config_path), '..', 'results', f'{model_type_global}_tuning_{timestamp}'))
    os.makedirs(tuning_results_dir, exist_ok=True)
    log_file_path = os.path.join(tuning_results_dir, 'tuning.log')
    
    log_level_str = config_global.get('logging', {}).get('log_level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    # 初始化日志记录器
    initialize_logger(log_file_path, log_level)
    logger.info(f"Logger setup complete. Log level: {log_level_str}. Log file: {log_file_path}")
    logger.info(f"Tuning results and artifacts will be saved in: {tuning_results_dir}")
    logger.info(f"Using configuration from: {config_path}")
    
    # --- Data Loading and Processing --- 
    # (Similar to train.py, but only process train data and fit preprocessors here)
    logger.info("--- Loading and Processing Training Data for Tuning ---")
    # ... [Data Loading] ...
    data_cfg = config_global.get('data', {})
    raw_data_dir = data_cfg.get('dataset_path', 'data/raw/PHM_2010/')
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '../..'))
    if not os.path.isabs(raw_data_dir):
        raw_data_dir = os.path.abspath(os.path.join(project_root, raw_data_dir))
    raw_data_dict = load_phm2010_data(raw_data_dir)
    if not raw_data_dict: return

    # ... [Preprocessing] ...
    processed_data = {}
    train_exp_ids = config_global.get('training', {}).get('train_experiments', ['c1', 'c4']) # Load only train experiments
    logger.info(f"Processing data for training experiments: {train_exp_ids}")
    for exp_name in train_exp_ids:
         if exp_name in raw_data_dict:
             logger.info(f"Preprocessing {exp_name}...")
             processed_df = preprocess_signals(raw_data_dict[exp_name], config_global)
             if processed_df is not None: processed_data[exp_name] = processed_df
             else: logger.warning(f"Preprocessing failed for {exp_name}.")
         else:
             logger.warning(f"Training experiment {exp_name} not found in loaded data.")
    if not processed_data: logger.error("Preprocessing failed for all training experiments."); return
    del raw_data_dict

    # ... [Feature Extraction] ...
    features_df_all, target_series_all = extract_features_from_data(processed_data, config_global)
    del processed_data
    if features_df_all is None or target_series_all is None: logger.error("Feature extraction failed."); return

    # Ensure only training data is used
    available_train_exp = features_df_all.index.get_level_values('experiment_id').unique().tolist()
    train_exp_ids = [eid for eid in train_exp_ids if eid in available_train_exp]
    if not train_exp_ids:
         logger.error("No training experiment data found after feature extraction."); return
    
    X_train_raw = features_df_all[features_df_all.index.get_level_values('experiment_id').isin(train_exp_ids)]
    y_train_global = target_series_all[target_series_all.index.get_level_values('experiment_id').isin(train_exp_ids)]
    del features_df_all, target_series_all
    
    if X_train_raw.empty or y_train_global.empty:
         logger.error("Training data is empty after filtering experiments."); return

    # ... [Feature Selection] ...
    # Apply selection based on the training data only
    logger.info("--- Selecting Features based on Training Data ---")
    X_train_selected = select_features(X_train_raw, y_train_global, config_global)
    if X_train_selected.empty or X_train_selected is None: logger.error("Feature selection failed."); return
    y_train_global = y_train_global.loc[X_train_selected.index] # Realign target
    del X_train_raw

    # ... [Fit Scaler and PCA on Training Data] ...
    logger.info("--- Fitting Scaler and PCA on Training Data ---")
    # Do not save models here, just fit them for use in objective function
    pca_result_train = apply_pca(X_train_selected, config_global, save_path=None)
    if pca_result_train is None:
        logger.error("Fitting PCA on training data failed.")
        return
    # Store the fitted preprocessors globally for the objective function
    # Note: apply_pca needs modification to return scaler/pca even if save_path is None
    # Temporary workaround: Call again with save path to get them, then delete files?
    temp_save_path = os.path.join(tuning_results_dir, 'temp_preprocessing')
    pca_result_with_models = apply_pca(X_train_selected, config_global, save_path=temp_save_path)
    if pca_result_with_models is None: logger.error("Failed to get fitted scaler/pca."); return
    X_train_global, scaler_global, pca_model_global = pca_result_with_models
    # Clean up temporary files
    try: 
        os.remove(os.path.join(temp_save_path, 'scaler.joblib'))
        os.remove(os.path.join(temp_save_path, 'pca.joblib'))
        os.rmdir(temp_save_path)
    except OSError as rm_e: logger.warning(f"Could not remove temp preprocessing files: {rm_e}")
        
    del X_train_selected
    logger.info(f"Processed training data shape for tuning: {X_train_global.shape}")


    # --- Define Search Space --- 
    search_space = []
    param_names = [] # Keep track of parameter names in order

    if model_type_global == 'random_forest':
        rf_cfg = config_global.get('models', {}).get('random_forest', {})
        rf_tuning_cfg = config_global.get('tuning', {}).get('random_forest', {})
        
        # 从tuning配置中获取参数范围，如果没有则使用默认值
        n_est_range = rf_tuning_cfg.get('n_estimators_range', [100, 500])
        max_depth_range = rf_tuning_cfg.get('max_depth_range', [5, 30])
        min_samples_split_range = rf_tuning_cfg.get('min_samples_split_range', [2, 10])
        min_samples_leaf_range = rf_tuning_cfg.get('min_samples_leaf_range', [1, 5])
        max_features_options = rf_tuning_cfg.get('max_features_options', ['sqrt', 'log2', 0.5])
        criterion_options = rf_tuning_cfg.get('criterion_options', ['absolute_error', 'squared_error'])
        
        # 添加所有搜索参数
        search_space.append(Integer(n_est_range[0], n_est_range[1], name='n_estimators'))
        search_space.append(Integer(max_depth_range[0], max_depth_range[1], name='max_depth'))
        search_space.append(Integer(min_samples_split_range[0], min_samples_split_range[1], name='min_samples_split'))
        search_space.append(Integer(min_samples_leaf_range[0], min_samples_leaf_range[1], name='min_samples_leaf'))
        
        # 处理max_features参数 - 可能包含字符串和数值
        if any(isinstance(opt, str) for opt in max_features_options) and any(isinstance(opt, (int, float)) for opt in max_features_options):
            logger.warning("max_features_options 混合了字符串和数值，仅使用字符串选项")
            str_options = [opt for opt in max_features_options if isinstance(opt, str)]
            search_space.append(Categorical(str_options, name='max_features'))
        elif all(isinstance(opt, str) for opt in max_features_options):
            search_space.append(Categorical(max_features_options, name='max_features'))
        elif all(isinstance(opt, (int, float)) for opt in max_features_options):
            min_val, max_val = min(max_features_options), max(max_features_options)
            search_space.append(Real(min_val, max_val, prior='uniform', name='max_features'))
        
        # 添加criterion参数
        search_space.append(Categorical(criterion_options, name='criterion'))
        
        param_names = [dim.name for dim in search_space]
        
    elif model_type_global == 'bpnn':
        bpnn_cfg = config_global.get('models', {}).get('bpnn', {})
        bpnn_tuning_cfg = config_global.get('tuning', {}).get('bpnn', {})
        
        # 从tuning配置中获取参数范围
        lr_range = bpnn_tuning_cfg.get('learning_rate_range', [1e-4, 1e-2])
        weight_decay_range = bpnn_tuning_cfg.get('weight_decay_range', [1e-4, 1e-2])
        dropout_rate_range = bpnn_tuning_cfg.get('dropout_rate_range', [0.1, 0.5])
        batch_size_options = bpnn_tuning_cfg.get('batch_size_options', [16, 32, 64])
        activation_options = bpnn_tuning_cfg.get('activation_options', ['relu', 'leaky_relu', 'tanh'])
        
        # 添加连续参数
        search_space.append(Real(lr_range[0], lr_range[1], prior='log-uniform', name='learning_rate'))
        search_space.append(Real(weight_decay_range[0], weight_decay_range[1], prior='log-uniform', name='weight_decay'))
        search_space.append(Real(dropout_rate_range[0], dropout_rate_range[1], prior='uniform', name='dropout_rate'))
        
        # 添加分类参数
        search_space.append(Categorical(batch_size_options, name='batch_size'))
        search_space.append(Categorical(activation_options, name='activation'))
        
        # 由于hidden_layers结构较为复杂，使用Categorical选择预定义的结构
        hidden_layers_options = bpnn_tuning_cfg.get('hidden_layers_options', [[64, 32], [128, 64], [64, 32, 16]])
        if hidden_layers_options:
            # 将列表转换为字符串表示，稍后在objective函数中解析回列表
            hidden_layers_str = [str(hl) for hl in hidden_layers_options]
            search_space.append(Categorical(hidden_layers_str, name='hidden_layers_str'))
        
        param_names = [dim.name for dim in search_space]
        
    else:
        logger.error(f"Tuning not implemented for model type: {model_type_global}")
        return

    if not search_space:
        logger.error("Search space is empty. Check configuration for tuning ranges.")
        return
        
    # Wrap objective function to accept named arguments from search space
    @use_named_args(search_space)
    def wrapped_objective(**params):
        return objective(**params)

    # --- Run Bayesian Optimization --- 
    tuning_cfg = config_global.get('tuning', {})
    n_calls = tuning_cfg.get('bayesian_optimization_iterations', 50)
    logger.info(f"Starting Bayesian Optimization with {n_calls} calls...")
    
    result = gp_minimize(
        func=wrapped_objective,
        dimensions=search_space,
        n_calls=n_calls,
        random_state=random_seed,
        # acq_func="EI", # Acquisition function (Expected Improvement)
        # n_initial_points=10 # Number of random points before Gaussian Process
    )

    # --- Results --- 
    best_params_list = result.x
    best_score = result.fun

    # Map list back to dictionary using param_names
    best_params_dict = {name: val for name, val in zip(param_names, best_params_list)}

    logger.info("--- Hyperparameter Tuning Finished ---")
    logger.info(f"Best Score ({tuning_cfg.get('optimization_metric', 'mae').upper()}): {best_score:.6f}")
    logger.info(f"Best Parameters: {best_params_dict}")

    # --- Save Best Parameters (Optional) ---
    try:
        best_params_path = os.path.join(tuning_results_dir, 'best_hyperparameters.yaml')
        with open(best_params_path, 'w') as f:
            yaml.dump(best_params_dict, f, default_flow_style=False)
        logger.info(f"Best hyperparameters saved to: {best_params_path}")
        
        # Optionally, update the main config file in memory with best params for potential immediate training?
        # config_global['models'][model_type_global].update(best_params_dict)
        # logger.info("Config in memory updated with best parameters.")
        
    except Exception as save_e:
        logger.error(f"Failed to save best hyperparameters: {save_e}")

# === Main Entry Point ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tune hyperparameters for a tool wear prediction model.")
    parser.add_argument('--config', type=str, required=True,
                        help="Path to the configuration YAML file.")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        exit(1)

    tune_hyperparameters(args.config)
