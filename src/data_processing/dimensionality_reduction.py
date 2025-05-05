import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import logging
import joblib # To save the scaler and PCA model
import os

logger = logging.getLogger(__name__)

def apply_pca(features_df, config, save_path=None):
    """对特征进行标准化和 PCA 降维。

    Args:
        features_df (pd.DataFrame): 经过特征选择后的特征矩阵。
        config (dict): 配置字典，包含 dimensionality_reduction 参数。
                       例如 config['dimensionality_reduction']['pca_n_components']。
        save_path (str, optional): 目录路径，用于保存拟合好的 StandardScaler 和 PCA 对象。
                                   如果提供，将保存 scaler.joblib 和 pca.joblib。
                                   Defaults to None (不保存).

    Returns:
        pd.DataFrame: 降维后的特征矩阵。列名将是 PC1, PC2, ...
        None: 如果发生错误。
        tuple(pd.DataFrame, StandardScaler, PCA): 如果提供了 save_path，则额外返回 scaler 和 pca 对象。
    """
    dr_config = config.get('dimensionality_reduction', {})
    n_components = dr_config.get('pca_n_components', 0.90) # 默认保留 90% 方差

    if features_df.empty:
        logger.error("Input features DataFrame is empty. Cannot apply PCA.")
        return None

    logger.info(f"Starting PCA dimensionality reduction. Target components/variance: {n_components}")
    logger.info(f"Original feature shape: {features_df.shape}")

    # 检查并处理 NaN/Inf 值
    if features_df.isnull().values.any() or np.isinf(features_df.values).any():
        logger.warning("NaN or Inf values detected in features before scaling. Replacing Inf with NaN and imputing NaNs using mean strategy.")
        # 1. Replace Inf with NaN
        features_no_inf = features_df.replace([np.inf, -np.inf], np.nan)
        
        # 2. Impute NaNs using SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        try:
            # fit_transform returns a numpy array, need to convert back to DataFrame
            # Keep track of original columns and index
            original_index_impute = features_no_inf.index
            original_columns_impute = features_no_inf.columns
            features_imputed_np = imputer.fit_transform(features_no_inf)
            features_imputed = pd.DataFrame(features_imputed_np, index=original_index_impute, columns=original_columns_impute)
            logger.info(f"Successfully imputed NaN values using mean strategy. Imputed features shape: {features_imputed.shape}")
        except Exception as impute_e:
             logger.error(f"Error during NaN imputation with SimpleImputer: {impute_e}", exc_info=True)
             return None # Stop if imputation fails
             
        # 3. Drop columns that might still be all NaN (if original column was all NaN/Inf)
        features_cleaned = features_imputed.dropna(axis=1, how='all')
        if features_cleaned.shape[1] < features_imputed.shape[1]:
             dropped_cols = features_imputed.columns.difference(features_cleaned.columns)
             logger.warning(f"Dropped {len(dropped_cols)} columns after imputation because they were still all NaN: {dropped_cols.tolist()}")
        
        if features_cleaned.empty:
             logger.error("All feature columns became NaN or were dropped after imputation. Cannot apply PCA.")
             return None
             
        features_to_process = features_cleaned
    else:
        features_to_process = features_df
        
    original_index = features_to_process.index # 保存原始索引
    original_columns = features_to_process.columns # 保存原始列名 (用于调试或理解)

    # 1. 标准化特征
    scaler = StandardScaler()
    try:
        scaled_features = scaler.fit_transform(features_to_process)
        logger.info("Features successfully scaled using StandardScaler.")
    except Exception as e:
        logger.error(f"Error during feature scaling: {e}", exc_info=True)
        return None

    # 2. 应用 PCA
    # 如果 n_components 是浮点数 (0~1)，PCA 会自动选择保留该比例方差所需的主成分数量
    # 如果是整数，则直接指定主成分数量
    if isinstance(n_components, float) and 0 < n_components < 1:
        logger.info(f"Applying PCA to retain {n_components*100:.1f}% of variance.")
    elif isinstance(n_components, int) and n_components > 0:
        logger.info(f"Applying PCA to get {n_components} principal components.")
        # 确保请求的组件数不超过可用特征数
        n_components = min(n_components, scaled_features.shape[1])
        logger.info(f"Adjusted n_components to {n_components} (cannot exceed number of features).")
    else:
        logger.error(f"Invalid value for pca_n_components: {n_components}. Must be int > 0 or float (0, 1).")
        return None

    pca = PCA(n_components=n_components)
    try:
        pca_features = pca.fit_transform(scaled_features)
        logger.info(f"PCA applied successfully. Reduced dimensions: {pca_features.shape}")
        logger.info(f"Explained variance ratio by selected components: {np.sum(pca.explained_variance_ratio_):.4f}")
        if isinstance(n_components, float):
             logger.info(f"Number of components selected to explain >= {n_components*100:.1f}% variance: {pca.n_components_}")
    except Exception as e:
        logger.error(f"Error during PCA fitting/transforming: {e}", exc_info=True)
        return None

    # 3. 创建降维后的 DataFrame
    pca_columns = [f'PC{i+1}' for i in range(pca_features.shape[1])]
    pca_df = pd.DataFrame(pca_features, columns=pca_columns, index=original_index)

    # 4. (可选) 保存 Scaler 和 PCA 模型
    if save_path:
        try:
            os.makedirs(save_path, exist_ok=True)
            scaler_path = os.path.join(save_path, 'scaler.joblib')
            pca_path = os.path.join(save_path, 'pca.joblib')
            joblib.dump(scaler, scaler_path)
            joblib.dump(pca, pca_path)
            logger.info(f"Scaler and PCA models saved to {save_path}")
            return pca_df, scaler, pca
        except Exception as e:
            logger.error(f"Error saving Scaler/PCA models to {save_path}: {e}")
            # 即使保存失败也返回结果
            return pca_df # Fallback to returning only the DataFrame

    return pca_df

# # --- Example Usage ---
# if __name__ == '__main__':
#     import yaml
#     import os
#     # Assume previous steps (loading, preprocessing, feature extraction, selection) are done
#     try:
#         from feature_extraction import extract_features_from_data
#         from feature_selection import select_features
#     except ImportError:
#         print("Make sure feature_extraction.py and feature_selection.py are accessible.")
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

#     # --- Mock Previous Steps (replace with actual pipeline calls) ---
#     # This part needs the output from feature_selection
#     # We'll create some dummy data for demonstration
#     print("Creating dummy selected features data for PCA example...")
#     # Ensure dummy data has multi-index like the real output
#     index = pd.MultiIndex.from_tuples([
#         ('c1', 1), ('c1', 2), ('c1', 3), ('c1', 4), ('c1', 5),
#         ('c4', 1), ('c4', 2), ('c4', 3), ('c4', 4), ('c4', 5)
#     ], names=['experiment_id', 'measurement_id'])
#     selected_features_df = pd.DataFrame(np.random.rand(10, 15), # 10 samples, 15 features
#                                         columns=[f'feat_{i}' for i in range(15)],
#                                         index=index)
#     print(f"Dummy selected features shape: {selected_features_df.shape}")
#     print(selected_features_df.head())
#     # --- End Mock --- 

#     # 2. Apply PCA
#     print("\nApplying PCA...")
#     # Example: Save models to a results directory
#     save_dir = '../../results/models/' # Relative path example
#     pca_result = apply_pca(selected_features_df, config, save_path=save_dir)

#     if pca_result is not None:
#         if save_path and isinstance(pca_result, tuple):
#              pca_df, saved_scaler, saved_pca = pca_result
#              print("PCA successful! Scaler and PCA models saved.")
#              print(f"Saved scaler type: {type(saved_scaler)}")
#              print(f"Saved PCA type: {type(saved_pca)}")
#         else:
#              pca_df = pca_result
#              print("PCA successful! Models not saved (save_path was None or saving failed).")
        
#         print("\nPCA reduced DataFrame shape:", pca_df.shape)
#         print("PCA reduced DataFrame head:")
#         print(pca_df.head())
        
#         # Example: Loading the saved models (if saved)
#         # if save_path:
#         #     try:
#         #         loaded_scaler = joblib.load(os.path.join(save_dir, 'scaler.joblib'))
#         #         loaded_pca = joblib.load(os.path.join(save_dir, 'pca.joblib'))
#         #         print("\nSuccessfully loaded saved Scaler and PCA models.")
#         #         # You could use loaded_scaler.transform() and loaded_pca.transform() on new data
#         #     except Exception as load_e:
#         #         print(f"Error loading saved models: {load_e}")

#     else:
#         print("\nPCA dimensionality reduction failed.")
