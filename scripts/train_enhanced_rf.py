#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化的随机森林训练脚本
包含特征选择、参数调优、交叉验证等高级功能
"""

import pandas as pd
import numpy as np
import yaml
import logging
import os
import argparse
import joblib
from datetime import datetime
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

# Import necessary functions from src
try:
    from src.models.random_forest_enhanced import (
        build_enhanced_random_forest, 
        build_forest_ensemble,
        select_features_with_importance,
        recursive_feature_elimination,
        random_forest_grid_search,
        create_rf_ensemble
    )
    from src.training.evaluate import calculate_regression_metrics, check_metrics_thresholds
    from src.utils.logger import setup_logger
except ImportError as e:
    raise ImportError(f"Could not import project modules. Ensure PYTHONPATH is set correctly or run from root: {e}")

def train_enhanced_rf(config_path):
    """优化版随机森林训练流程

    Args:
        config_path (str): 配置文件路径
    """
    # 1. 加载配置
    print(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        return

    # --- 创建结果目录 ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.abspath(os.path.join(project_root, 'results', f'rf_enhanced_{timestamp}'))
    os.makedirs(results_dir, exist_ok=True)
    viz_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    log_file_path = os.path.join(results_dir, 'rf_enhanced.log')

    # --- 配置日志 ---
    log_level_str = config.get('logging', {}).get('log_level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger = setup_logger('RFEnhancedLogger', log_level, log_file_path)
    logger.info(f"Logger setup complete. Log level: {log_level_str}. Log file: {log_file_path}")
    logger.info(f"Results will be saved in: {results_dir}")
    logger.info(f"Using configuration from: {config_path}")

    # --- 设置随机种子 ---
    random_seed = config.get('training', {}).get('random_state', 42)
    logger.info(f"Setting random seed to: {random_seed}")
    np.random.seed(random_seed)

    # --- 获取数据路径 ---
    processed_data_dir_rel = config.get('data', {}).get('processed_data_dir', 'data/processed')
    processed_data_dir = os.path.abspath(os.path.join(project_root, processed_data_dir_rel))
    features_path = os.path.join(processed_data_dir, 'final_features.csv')
    target_path = os.path.join(processed_data_dir, 'target_wear.csv')

    # 2. 加载特征和目标
    logger.info("--- Step 1: Loading Data ---")
    try:
        # 加载特征
        final_features_df = pd.read_csv(features_path, index_col=['experiment_id', 'measurement_id'])
        # 加载目标
        target_series = pd.read_csv(target_path, index_col=['experiment_id', 'measurement_id']).squeeze("columns")
        if not isinstance(target_series, pd.Series):
             raise ValueError("Loaded target is not a Pandas Series after squeeze.")
        logger.info(f"Loaded features shape: {final_features_df.shape}")
        logger.info(f"Loaded target shape: {target_series.shape}")
    except FileNotFoundError:
        logger.error(f"Features file or target file not found. Run engineer_features.py first.")
        return
    except Exception as e:
        logger.error(f"Error loading features or target: {e}", exc_info=True)
        return
        
    # 检查索引对齐
    if not final_features_df.index.equals(target_series.index):
         logger.warning("Index mismatch between loaded features and target. Attempting to align.")
         common_index = final_features_df.index.intersection(target_series.index)
         if common_index.empty:
              logger.error("No common index found between features and target after loading. Aborting.")
              return
         final_features_df = final_features_df.loc[common_index]
         target_series = target_series.loc[common_index]
         logger.info(f"Aligned data shapes: Features={final_features_df.shape}, Target={target_series.shape}")

    # 3. 划分数据集
    logger.info("--- Step 2: Splitting Data ---")
    train_exp_ids = config.get('training', {}).get('train_experiments', ['c1', 'c4'])
    test_exp_ids = config.get('training', {}).get('test_experiments', ['c6'])

    available_exp = final_features_df.index.get_level_values('experiment_id').unique().tolist()
    train_exp_ids = [eid for eid in train_exp_ids if eid in available_exp]
    test_exp_ids = [eid for eid in test_exp_ids if eid in available_exp]

    if not train_exp_ids:
        logger.error(f"No training experiments found in the data. Aborting.")
        return
    if not test_exp_ids:
        logger.warning(f"No testing experiments found. Evaluation will be skipped.")

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

    # 4. 特征选择与工程
    logger.info("--- Step 3: Feature Selection ---")

    # 创建基础随机森林模型
    base_rf = build_enhanced_random_forest(config, random_state=random_seed)
    if base_rf is None:
        logger.error("Failed to build base RF model. Aborting.")
        return

    # 使用特征重要性进行选择
    logger.info("Performing feature selection using feature importance...")
    # 从配置文件中获取特征重要性阈值
    importance_threshold = config.get('feature_selection', {}).get('importance_threshold', 0.01)
    X_train_selected, selected_features, importance_df = select_features_with_importance(
        base_rf, X_train, y_train, threshold=importance_threshold, results_dir=results_dir
    )
    
    # 如果有测试集，应用相同的特征选择
    if not X_test.empty:
        X_test_selected = X_test[selected_features]
        logger.info(f"Applied feature selection to test set. New shape: {X_test_selected.shape}")
    else:
        X_test_selected = X_test
    
    # 添加特征缩放
    logger.info("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_selected),
        index=X_train.index,
        columns=selected_features
    )
    
    if not X_test.empty:
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test_selected),
            index=X_test.index,
            columns=selected_features
        )
    else:
        X_test_scaled = X_test_selected

    # 保存特征选择和缩放模型
    joblib.dump(scaler, os.path.join(results_dir, 'feature_scaler.joblib'))
    
    # 5. 检查是否使用集成学习
    ensemble_enabled = config.get('models', {}).get('ensemble', {}).get('enabled', False)
    
    if ensemble_enabled:
        logger.info("--- Step 4: Building Ensemble Model ---")
        # 创建随机森林集成模型
        best_model = build_forest_ensemble(config, random_state=random_seed)
        logger.info("Training ensemble model...")
        best_model.fit(X_train_scaled, y_train)
        
        # 保存模型
        joblib.dump(best_model, os.path.join(results_dir, 'rf_ensemble_model.joblib'))
        logger.info(f"Ensemble model saved to {os.path.join(results_dir, 'rf_ensemble_model.joblib')}")
        
        # 在训练集上评估
        y_train_pred = best_model.predict(X_train_scaled)
        train_metrics = calculate_regression_metrics(y_train, y_train_pred)
        logger.info(f"Training set metrics for ensemble: {train_metrics}")
        
        # 在测试集上评估
        if not X_test.empty and not y_test.empty:
            logger.info("Evaluating ensemble on test set...")
            y_test_pred = best_model.predict(X_test_scaled)
            test_metrics = calculate_regression_metrics(y_test, y_test_pred)
            logger.info(f"Test set metrics for ensemble: {test_metrics}")
            
            # 检查阈值
            thresholds = config.get('training', {}).get('target_metrics', {})
            passed_thresholds = check_metrics_thresholds(test_metrics, thresholds)
            logger.info(f"Threshold check result for ensemble: {passed_thresholds}")
            test_metrics['passed_thresholds'] = passed_thresholds
            
            # 保存评估结果
            with open(os.path.join(results_dir, 'ensemble_evaluation_results.yaml'), 'w') as f:
                yaml.dump(test_metrics, f, default_flow_style=False)
            
            # 保存预测结果
            predictions_df = pd.DataFrame({
                'actual': y_test,
                'predicted': y_test_pred
            }, index=X_test.index)
            predictions_df.to_csv(os.path.join(results_dir, 'ensemble_test_predictions.csv'))
            
            # 绘制预测与实际值对比图
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(y_test)), y_test.values, 'b-', label='Actual')
            plt.plot(range(len(y_test_pred)), y_test_pred, 'r--', label='Ensemble Predicted')
            plt.xlabel('Sample')
            plt.ylabel('Tool Wear')
            plt.title('Ensemble Predictions vs Actual Values')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'ensemble_predictions_vs_actual.png'))
            plt.close()
            
            # 绘制散点图
            plt.figure(figsize=(10, 10))
            plt.scatter(y_test, y_test_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
            plt.xlabel('Actual Tool Wear')
            plt.ylabel('Predicted Tool Wear')
            plt.title('Ensemble Prediction Scatter Plot')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'ensemble_prediction_scatter.png'))
            plt.close()
    else:
        # 6. 超参数优化
        logger.info("--- Step 4: Hyperparameter Optimization ---")
        
        # 从配置文件中获取参数网格
        grid_search_config = config.get('grid_search', {})
        grid_search_enabled = grid_search_config.get('enabled', False)
        
        if grid_search_enabled:
            # 读取配置文件中的参数网格
            param_grid = grid_search_config.get('param_grid', {})
            cv_folds = grid_search_config.get('cv_folds', 5)
            scoring = grid_search_config.get('scoring', 'neg_mean_absolute_error')
            n_jobs = grid_search_config.get('n_jobs', -1)
            
            logger.info(f"使用从配置文件加载的网格搜索参数：")
            for param, values in param_grid.items():
                logger.info(f"  - {param}: {values}")
            logger.info(f"网格搜索设置：CV折数={cv_folds}, 评分={scoring}, 并行任务数={n_jobs}")
            
            # 计算总网格组合数
            import math
            grid_size = 1
            for values in param_grid.values():
                grid_size *= len(values)
            logger.info(f"总网格组合数：{grid_size}，预计将运行 {grid_size * cv_folds} 次拟合")
        else:
            # 使用默认参数网格
            logger.info("网格搜索配置未启用，使用默认参数网格")
            param_grid = {
                'n_estimators': [300, 500, 700],
                'max_depth': [15, 20, 25, None],
                'min_samples_split': [2, 3, 5],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.33]
            }
            cv_folds = 5
            scoring = 'neg_mean_absolute_error'
            n_jobs = -1
        
        # 执行网格搜索
        logger.info(f"开始网格搜索，这可能需要较长时间...")
        start_time = datetime.now()
        best_model = random_forest_grid_search(
            X_train_scaled, y_train, param_grid, cv=cv_folds, n_jobs=n_jobs, 
            results_dir=results_dir, scoring=scoring
        )
        end_time = datetime.now()
        search_duration = end_time - start_time
        logger.info(f"网格搜索完成，耗时：{search_duration}")
        
        if best_model is None:
            logger.warning("网格搜索失败。使用默认随机森林模型作为替代。")
            best_model = build_enhanced_random_forest(config, random_state=random_seed)
    
    # 7. 模型训练与评估
    logger.info("--- Step 5: Model Training and Evaluation ---")
    logger.info("Training final model...")
    
    # 如果没有使用集成学习，则此时需要训练模型
    if not ensemble_enabled:
        best_model.fit(X_train_scaled, y_train)
        
        # 保存模型
        joblib.dump(best_model, os.path.join(results_dir, 'rf_final_model.joblib'))
        logger.info(f"Final RF model saved to {os.path.join(results_dir, 'rf_final_model.joblib')}")
        
        # 在训练集上评估
        y_train_pred = best_model.predict(X_train_scaled)
        train_metrics = calculate_regression_metrics(y_train, y_train_pred)
        logger.info(f"Training set metrics: {train_metrics}")
        
        # 在测试集上评估
        if not X_test.empty and not y_test.empty:
            logger.info("Evaluating on test set...")
            y_test_pred = best_model.predict(X_test_scaled)
            test_metrics = calculate_regression_metrics(y_test, y_test_pred)
            logger.info(f"Test set metrics: {test_metrics}")
            
            # 检查阈值
            thresholds = config.get('training', {}).get('target_metrics', {})
            passed_thresholds = check_metrics_thresholds(test_metrics, thresholds)
            logger.info(f"Threshold check result: {passed_thresholds}")
            test_metrics['passed_thresholds'] = passed_thresholds
            
            # 保存评估结果
            with open(os.path.join(results_dir, 'evaluation_results.yaml'), 'w') as f:
                yaml.dump(test_metrics, f, default_flow_style=False)
            
            # 保存预测结果
            predictions_df = pd.DataFrame({
                'actual': y_test,
                'predicted': y_test_pred
            }, index=X_test.index)
            predictions_df.to_csv(os.path.join(results_dir, 'test_predictions.csv'))
            
            # 绘制预测与实际值对比图
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(y_test)), y_test.values, 'b-', label='Actual')
            plt.plot(range(len(y_test_pred)), y_test_pred, 'r--', label='Predicted')
            plt.xlabel('Sample')
            plt.ylabel('Tool Wear')
            plt.title('Random Forest Predictions vs Actual Values')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'predictions_vs_actual.png'))
            plt.close()
            
            # 绘制散点图
            plt.figure(figsize=(10, 10))
            plt.scatter(y_test, y_test_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
            plt.xlabel('Actual Tool Wear')
            plt.ylabel('Predicted Tool Wear')
            plt.title('Random Forest Prediction Scatter Plot')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'prediction_scatter.png'))
            plt.close()
    
    # 8. 特征重要性可视化
    logger.info("--- Step 6: Feature Importance Visualization ---")
    
    # 提取特征重要性
    if hasattr(best_model, 'feature_importances_'):
        feature_importances = best_model.feature_importances_
    elif hasattr(best_model, 'estimators_') and len(best_model.estimators_) > 0:
        # 对于集成模型，尝试从第一个基础模型中获取特征重要性
        try:
            if hasattr(best_model, 'named_estimators_'):
                # Stacking/Voting Regressor
                first_estimator = list(best_model.named_estimators_.values())[0]
            else:
                # 其他类型的集成
                first_estimator = best_model.estimators_[0]
                
            if hasattr(first_estimator, 'feature_importances_'):
                feature_importances = first_estimator.feature_importances_
            else:
                logger.warning("Could not extract feature importances from ensemble model.")
                feature_importances = None
        except Exception as e:
            logger.warning(f"Error extracting feature importances: {e}")
            feature_importances = None
    else:
        logger.warning("Model does not have feature_importances_ attribute.")
        feature_importances = None
    
    # 如果成功提取到特征重要性，则进行可视化
    if feature_importances is not None:
        # 创建特征重要性数据框
        importance_df = pd.DataFrame({
            'feature': X_train_scaled.columns,
            'importance': feature_importances
        }).sort_values('importance', ascending=False)
        
        # 保存特征重要性
        importance_df.to_csv(os.path.join(results_dir, 'final_feature_importance.csv'), index=False)
        
        # 绘制特征重要性条形图
        plt.figure(figsize=(12, 8))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'final_feature_importance.png'))
        plt.close()
    
    # 9. 保存使用的配置
    with open(os.path.join(results_dir, 'config_used.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info("--- Enhanced Random Forest Training Completed ---")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate Enhanced Random Forest Model")
    parser.add_argument('--config', type=str, default='config/rf_optimized.yaml',
                        help='Path to the configuration file (default: config/rf_optimized.yaml)')
    args = parser.parse_args()

    if not os.path.isabs(args.config):
        config_file_path = os.path.join(project_root, args.config)
    else:
        config_file_path = args.config

    if not os.path.exists(config_file_path):
        print(f"Error: Config file not found at {config_file_path}")
        sys.exit(1)

    train_enhanced_rf(config_file_path) 