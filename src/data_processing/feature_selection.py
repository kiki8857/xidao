import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import logging

logger = logging.getLogger(__name__)

def calculate_correlations(features_df, target_series):
    """计算特征与目标之间的 Spearman 相关系数。

    Args:
        features_df (pd.DataFrame): 特征矩阵。
        target_series (pd.Series): 目标变量 (例如 'wear')。

    Returns:
        pd.Series: 每个特征与目标之间的 Spearman 相关系数绝对值。
                  索引是特征名，值是相关系数绝对值。返回空 Series 如果出错。
    """
    correlations = {}
    if features_df.empty or target_series.empty or len(features_df) != len(target_series):
        logger.error("Features or target are empty or have mismatched lengths.")
        return pd.Series(dtype=float)

    # 确保 target_series 和 features_df 的索引一致，以便计算
    common_index = features_df.index.intersection(target_series.index)
    if len(common_index) < 2: # Need at least 2 points for correlation
        logger.error("Not enough common data points between features and target after aligning index.")
        return pd.Series(dtype=float)
    
    aligned_features = features_df.loc[common_index]
    aligned_target = target_series.loc[common_index]

    # 处理可能存在的 NaN 或 Inf 值
    aligned_features = aligned_features.replace([np.inf, -np.inf], np.nan).fillna(aligned_features.mean()) # 简单填充均值
    aligned_target = aligned_target.replace([np.inf, -np.inf], np.nan).fillna(aligned_target.mean())
    # 删除仍然全为 NaN 的列
    aligned_features = aligned_features.dropna(axis=1, how='all')
    if aligned_features.empty:
         logger.error("All feature columns are NaN after cleaning.")
         return pd.Series(dtype=float)
    # 如果 target 全是 NaN
    if aligned_target.isnull().all():
         logger.error("Target series is all NaN after cleaning.")
         return pd.Series(dtype=float)


    for feature_name in aligned_features.columns:
        try:
            # 计算 Spearman 相关系数和 p-value
            corr, p_value = spearmanr(aligned_features[feature_name], aligned_target, nan_policy='omit')
            # 如果结果是 NaN (可能因为标准差为0), 则设为0
            correlations[feature_name] = abs(corr) if not np.isnan(corr) else 0
        except Exception as e:
            logger.warning(f"Could not calculate Spearman correlation for feature '{feature_name}': {e}")
            correlations[feature_name] = 0 # Or np.nan?

    return pd.Series(correlations)

def remove_redundant_features(features_df, threshold=0.9):
    """使用 Pearson 相关系数移除高度冗余的特征。

    Args:
        features_df (pd.DataFrame): 特征矩阵。
        threshold (float): Pearson 相关系数绝对值的阈值，高于此阈值的特征对将被视为冗余。

    Returns:
        pd.DataFrame: 移除了冗余特征后的 DataFrame。
    """
    # 处理 NaN/Inf
    df = features_df.replace([np.inf, -np.inf], np.nan).fillna(features_df.mean())
    df = df.dropna(axis=1, how='all') # 删除全 NaN 的列
    if df.empty:
        return df
        
    corr_matrix = df.corr(method='pearson').abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # 找到需要移除的特征列
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

    if to_drop:
        logger.info(f"Removing {len(to_drop)} redundant features based on Pearson correlation > {threshold}: {to_drop}")
        df_reduced = df.drop(columns=to_drop)
    else:
        logger.info(f"No redundant features found with Pearson correlation > {threshold}.")
        df_reduced = df
        
    # 确保返回的列是原始 DataFrame 中的子集 (避免因填充引入新列?)
    final_columns = [col for col in df_reduced.columns if col in features_df.columns]
    return features_df[final_columns].copy() # 返回原始数据的子集

def select_features(features_df, target_series, config):
    """根据相关性和冗余度筛选特征。

    Args:
        features_df (pd.DataFrame): 原始特征矩阵 (来自 feature_extraction)。
        target_series (pd.Series): 目标变量。
        config (dict): 配置字典，包含 feature_selection 参数。

    Returns:
        pd.DataFrame: 筛选后的特征矩阵。返回原始矩阵如果出错或未配置筛选。
    """
    fs_config = config.get('feature_selection', {})
    spearman_thresh = fs_config.get('spearman_threshold', 0.8)
    pearson_redundancy_thresh = fs_config.get('pearson_threshold', 0.8) # 用于冗余特征
    # n_features = fs_config.get('n_features_to_select', None) # 暂时不使用固定数量

    if not fs_config or not spearman_thresh:
        logger.info("Feature selection based on correlation is not configured or threshold is zero. Returning original features.")
        return features_df

    logger.info(f"Starting feature selection process...")
    logger.info(f"Initial number of features: {features_df.shape[1]}")

    # 1. 基于 Spearman 相关性筛选 (与目标的单调性)
    spearman_correlations = calculate_correlations(features_df, target_series)
    if spearman_correlations.empty:
        logger.error("Could not calculate Spearman correlations. Returning original features.")
        return features_df

    features_to_keep_spearman = spearman_correlations[spearman_correlations >= spearman_thresh].index.tolist()

    if not features_to_keep_spearman:
        logger.warning(f"No features met the Spearman correlation threshold >= {spearman_thresh}. Returning all features (or consider lowering threshold).")
        # 返回原始特征还是空 DataFrame? 暂时返回原始
        return features_df
    else:
        logger.info(f"{len(features_to_keep_spearman)} features selected based on Spearman correlation >= {spearman_thresh}."
                    f" Example kept: {features_to_keep_spearman[:5]}...")
        selected_df = features_df[features_to_keep_spearman].copy()

    # 2. 基于 Pearson 相关性移除冗余特征 (特征之间的共线性)
    if pearson_redundancy_thresh and pearson_redundancy_thresh < 1.0:
         logger.info(f"Removing redundant features with Pearson correlation > {pearson_redundancy_thresh}...")
         selected_df = remove_redundant_features(selected_df, threshold=pearson_redundancy_thresh)
    else:
        logger.info("Skipping redundant feature removal based on Pearson correlation (threshold not set or >= 1.0).")


    # 3. (可选) 按数量选择 (如果需要，通常在降维步骤前使用)
    # if n_features and selected_df.shape[1] > n_features:
    #     # 可以基于之前的相关性排序选择 top N
    #     top_features = spearman_correlations.loc[selected_df.columns].sort_values(ascending=False).head(n_features).index.tolist()
    #     selected_df = selected_df[top_features]
    #     logger.info(f"Selected top {n_features} features based on Spearman correlation.")

    final_feature_count = selected_df.shape[1]
    # 对比 config 中的目标数量 (10-15)
    target_n_features = fs_config.get('n_features_to_select', 12) # 从 config 获取目标
    if isinstance(target_n_features, int) and 10 <= target_n_features <= 15:
        if not (10 <= final_feature_count <= 15):
             logger.warning(f"Final number of selected features ({final_feature_count}) is outside the target range (10-15 specified in 开题/config). Consider adjusting thresholds.")
    else:
        logger.info(f"Config target n_features_to_select ({target_n_features}) is not within 10-15 range. Skipping range check.")


    logger.info(f"Feature selection complete. Final number of features: {final_feature_count}")
    logger.debug(f"Selected features: {selected_df.columns.tolist()}")
    return selected_df

# # --- Example Usage ---
# if __name__ == '__main__':
#     import yaml
#     import os
#     # 假设 feature_extraction.py 在同级目录或已安装
#     try:
#         from feature_extraction import extract_features_from_data
#     except ImportError:
#         print("Make sure feature_extraction.py is accessible.")
#         exit()
#     from loader import load_phm2010_data
#     from preprocessing import preprocess_signals
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#     # 1. Load Config
#     config_path = '../../config/config.yaml'
#     try:
#         with open(config_path, 'r') as f:
#             config = yaml.safe_load(f)
#     except Exception as e:
#         print(f"Error loading config: {e}")
#         config = {}

#     # 2. Load Data
#     script_dir = os.path.dirname(__file__)
#     project_root = os.path.abspath(os.path.join(script_dir, '../../'))
#     root_data_dir = config.get('data', {}).get('dataset_path', 'data/raw/PHM_2010/')
#     root_data_dir_abs = os.path.abspath(os.path.join(project_root, root_data_dir))
#     raw_data_dict = load_phm2010_data(root_data_dir_abs)

#     if not raw_data_dict:
#          print("Failed to load raw data.")
#          exit()

#     # 3. Preprocess Data (Example: only c1)
#     processed_data = {}
#     exp_to_process = 'c1'
#     if exp_to_process in raw_data_dict:
#         print(f"\nPreprocessing {exp_to_process} data...")
#         processed_data[exp_to_process] = preprocess_signals(raw_data_dict[exp_to_process], config)
#         print(f"Finished preprocessing {exp_to_process}.")
#     else:
#         print(f"Experiment {exp_to_process} not found.")
#         exit()
#     if not processed_data or processed_data[exp_to_process] is None:
#          print(f"Preprocessing failed for {exp_to_process}.")
#          exit()

#     # 4. Extract Features
#     print(f"\nExtracting features from preprocessed {exp_to_process} data...")
#     features_df, wear_series = extract_features_from_data(processed_data, config)

#     if features_df is None or wear_series is None:
#         print("Feature extraction failed.")
#         exit()

#     print(f"\nOriginal features shape: {features_df.shape}")

#     # 5. Select Features
#     print("\nSelecting features...")
#     selected_features_df = select_features(features_df, wear_series, config)

#     if selected_features_df is not None:
#         print("\nFeature selection successful!")
#         print("Selected features DataFrame shape:", selected_features_df.shape)
#         print("Selected features columns:")
#         print(selected_features_df.columns.tolist())
#     else:
#         print("\nFeature selection failed or returned None.")


