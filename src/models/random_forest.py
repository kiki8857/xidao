from sklearn.ensemble import RandomForestRegressor
import logging

logger = logging.getLogger(__name__)

def build_random_forest(config, random_state=None):
    """构建一个随机森林回归模型。

    Args:
        config (dict): 配置字典，应包含 models.random_forest 下的参数，
                       例如 n_estimators, max_depth, 等。
                       预期参数名与 RandomForestRegressor 的参数匹配。
        random_state (int, optional): 用于复现的随机种子。

    Returns:
        RandomForestRegressor: 一个 scikit-learn 随机森林回归模型实例。
        None: 如果配置错误。
    """
    rf_config = config.get('models', {}).get('random_forest', {})

    # 提取 RandomForestRegressor 支持的参数
    # 注意：移除了范围参数，训练/调优脚本会处理范围
    n_estimators = rf_config.get('n_estimators', 100) # 提供默认值
    max_depth = rf_config.get('max_depth', None)
    min_samples_split = rf_config.get('min_samples_split', 2)
    min_samples_leaf = rf_config.get('min_samples_leaf', 1)
    max_features = rf_config.get('max_features', 1.0) # scikit-learn >= 1.1 defaults to 1.0, older was 'auto'(sqrt)
    # 可以添加更多参数如 criterion, bootstrap, oob_score 等

    # 检查 n_estimators_range 和 max_depth_range 是否存在（这些用于调优，不用于构建）
    if 'n_estimators_range' in rf_config or 'max_depth_range' in rf_config:
         logger.warning("'n_estimators_range' and 'max_depth_range' are for tuning, using specific values (or defaults) for building the model.")

    logger.info(f"Building RandomForestRegressor with parameters: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, max_features={max_features}, random_state={random_state}")

    try:
        model = RandomForestRegressor(
            n_estimators=int(n_estimators), # 确保是整数
            max_depth=max_depth if max_depth is None else int(max_depth),
            min_samples_split=int(min_samples_split),
            min_samples_leaf=int(min_samples_leaf),
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1 # 使用所有可用 CPU 核
            # Add other parameters here if needed
        )
        return model
    except Exception as e:
        logger.error(f"Failed to build RandomForestRegressor: {e}", exc_info=True)
        return None

# # --- Example Usage ---
# if __name__ == '__main__':
#     import yaml
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#     # 1. Load Config
#     config_path = '../../config/config.yaml' # Adjust path as needed
#     try:
#         with open(config_path, 'r') as f:
#             config = yaml.safe_load(f)
#     except Exception as e:
#         print(f"Error loading config: {e}")
#         config = {}

#     # 2. Build Model
#     print("Building Random Forest model...")
#     rf_model = build_random_forest(config, random_state=42)

#     if rf_model:
#         print("\nRandom Forest model built successfully!")
#         print("Model parameters:")
#         print(rf_model.get_params())
#     else:
#         print("\nFailed to build Random Forest model.")
