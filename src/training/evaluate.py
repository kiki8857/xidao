import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)

def calculate_regression_metrics(y_true, y_pred):
    """计算回归任务的常用评估指标。

    Args:
        y_true (array-like): 真实的标签值。
        y_pred (array-like): 模型预测的标签值。

    Returns:
        dict: 包含 MAE, RMSE, R² 的字典。
              如果输入无效则返回空字典。
    """
    metrics = {}
    try:
        # 确保输入是 numpy array 或类似结构，并且非空
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if y_true.shape != y_pred.shape:
            logger.error(f"Shape mismatch between y_true ({y_true.shape}) and y_pred ({y_pred.shape}).")
            return {}
        if len(y_true) == 0:
            logger.warning("Input arrays y_true and y_pred are empty.")
            return {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan}
        
        # 检查 NaN 值 (可以根据需要决定如何处理)
        if np.isnan(y_true).any() or np.isnan(y_pred).any():
            logger.warning("NaN values found in y_true or y_pred. Metrics might be affected or NaN.")
            #可以选择移除含NaN的样本对再计算，或直接计算（结果可能为NaN）
            # mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
            # y_true = y_true[mask]
            # y_pred = y_pred[mask]
            # if len(y_true) == 0: return {} # 如果移除后为空

        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['r2'] = r2_score(y_true, y_pred)

        logger.debug(f"Calculated metrics: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")
        return metrics

    except Exception as e:
        logger.error(f"Error calculating regression metrics: {e}", exc_info=True)
        return {}

def check_metrics_thresholds(metrics, target_metrics_config):
    """检查计算出的指标是否满足配置文件中定义的阈值。

    Args:
        metrics (dict): 包含计算出的指标的字典 (例如来自 calculate_regression_metrics)。
        target_metrics_config (dict): 从主配置文件加载的目标指标阈值。
                                      例如: {'mae_threshold': 5.0, 'rmse_threshold': 5.0, 'r2_threshold': 0.85}

    Returns:
        bool: 如果所有定义的阈值都满足，则为 True，否则为 False。
    """
    if not metrics or not target_metrics_config:
        logger.warning("Metrics or target thresholds config is empty, cannot check thresholds.")
        return False # 或者 True? 取决于期望行为

    all_thresholds_met = True
    logger.info("Checking metrics against target thresholds...")

    if 'mae' in metrics and 'mae_threshold' in target_metrics_config:
        threshold = target_metrics_config['mae_threshold']
        if metrics['mae'] > threshold:
            logger.warning(f"MAE threshold NOT met: {metrics['mae']:.4f} > {threshold}")
            all_thresholds_met = False
        else:
             logger.info(f"MAE threshold met: {metrics['mae']:.4f} <= {threshold}")

    if 'rmse' in metrics and 'rmse_threshold' in target_metrics_config:
        threshold = target_metrics_config['rmse_threshold']
        if metrics['rmse'] > threshold:
            logger.warning(f"RMSE threshold NOT met: {metrics['rmse']:.4f} > {threshold}")
            all_thresholds_met = False
        else:
             logger.info(f"RMSE threshold met: {metrics['rmse']:.4f} <= {threshold}")

    if 'r2' in metrics and 'r2_threshold' in target_metrics_config:
        threshold = target_metrics_config['r2_threshold']
        if metrics['r2'] < threshold:
            logger.warning(f"R2 threshold NOT met: {metrics['r2']:.4f} < {threshold}")
            all_thresholds_met = False
        else:
             logger.info(f"R2 threshold met: {metrics['r2']:.4f} >= {threshold}")

    if all_thresholds_met:
        logger.info("All defined performance thresholds met!")
    else:
        logger.warning("One or more performance thresholds were NOT met.")
        
    return all_thresholds_met

# # --- Example Usage ---
# if __name__ == '__main__':
#     import yaml
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#     # 1. Load Target Metrics Config
#     config_path = '../../config/config.yaml' # Adjust path
#     target_metrics = {}
#     try:
#         with open(config_path, 'r') as f:
#             config = yaml.safe_load(f)
#             target_metrics = config.get('training', {}).get('target_metrics', {})
#             print(f"Loaded target metrics from config: {target_metrics}")
#     except Exception as e:
#         print(f"Error loading config or target metrics: {e}")

#     # 2. Example Data
#     y_true_example = np.array([10, 20, 30, 40, 50, 60])
#     y_pred_good = np.array([11, 19, 32, 38, 53, 58]) # Good predictions
#     y_pred_bad = np.array([15, 28, 25, 50, 45, 70]) # Bad predictions
#     y_pred_perfect = np.array([10, 20, 30, 40, 50, 60]) # Perfect predictions
#     y_pred_nan = np.array([11, 19, np.nan, 38, 53, 58]) # With NaN

#     # 3. Calculate Metrics
#     print("\n--- Calculating metrics for GOOD predictions ---")
#     metrics_good = calculate_regression_metrics(y_true_example, y_pred_good)
#     print(f"Metrics (Good): {metrics_good}")
#     check_metrics_thresholds(metrics_good, target_metrics)
    
#     print("\n--- Calculating metrics for BAD predictions ---")
#     metrics_bad = calculate_regression_metrics(y_true_example, y_pred_bad)
#     print(f"Metrics (Bad): {metrics_bad}")
#     check_metrics_thresholds(metrics_bad, target_metrics)

#     print("\n--- Calculating metrics for PERFECT predictions ---")
#     metrics_perfect = calculate_regression_metrics(y_true_example, y_pred_perfect)
#     print(f"Metrics (Perfect): {metrics_perfect}")
#     check_metrics_thresholds(metrics_perfect, target_metrics)
    
#     print("\n--- Calculating metrics with NaN prediction ---")
#     metrics_nan = calculate_regression_metrics(y_true_example, y_pred_nan)
#     print(f"Metrics (NaN): {metrics_nan}")
#     check_metrics_thresholds(metrics_nan, target_metrics)

#     print("\n--- Calculating metrics with empty input ---")
#     metrics_empty = calculate_regression_metrics([], [])
#     print(f"Metrics (Empty): {metrics_empty}")
#     check_metrics_thresholds(metrics_empty, target_metrics)
