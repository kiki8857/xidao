import numpy as np
import pandas as pd
import pywt # PyWavelets for wavelet denoising
from scipy.signal import savgol_filter # Savitzky-Golay for smoothing (alternative to moving avg)
import logging
from scipy import stats as sp_stats
from statsmodels.robust import mad as sm_mad # Import MAD from statsmodels

logger = logging.getLogger(__name__)

def remove_outliers_iqr(data_series, factor=1.5):
    """使用 IQR 方法移除异常值。"""
    q1 = data_series.quantile(0.25)
    q3 = data_series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    # 用 NaN 标记异常值，或用边界值替换
    # 这里使用 NaN 标记，后续可以填充或删除
    return data_series.where((data_series >= lower_bound) & (data_series <= upper_bound), np.nan)

def remove_outliers_zscore(data_series, threshold=3.0):
    """
    使用 Z-score 方法移除异常值。
    
    Args:
        data_series (pd.Series): 输入的数据序列
        threshold (float): Z-score阈值，默认为3.0
        
    Returns:
        pd.Series: 处理后的数据序列，异常值被替换为NaN
    """
    # 计算Z-scores
    z_scores = (data_series - data_series.mean()) / data_series.std()
    # 标记异常值为NaN
    return data_series.where(z_scores.abs() <= threshold, np.nan)

def wavelet_denoise(signal, wavelet='db4', level=1):
    """
    Apply wavelet denoising to a signal.
    """
    original_index = signal.index
    signal_clean = signal.dropna()
    if signal_clean.empty:
        logger.warning("Signal is empty after dropping NaNs, returning original signal (NaN filled).")
        # Return a Series with the original index filled with NaN
        return pd.Series(np.nan, index=original_index, dtype=float)

    try:
        # Decompose to get the wavelet coefficients
        coeffs = pywt.wavedec(signal_clean.values, wavelet, level=level) # Use .values for numpy array

        # Check if coeffs has enough elements. wavedec returns a list [cA_n, cD_n, cD_n-1, ..., cD_1]
        if len(coeffs) < 2:
            logger.warning(f"Wavelet decomposition did not produce enough coefficients (level={level}). Returning original signal.")
            # Return original signal reindexed
            return signal.reindex(original_index)

        # Extract detail coefficients (list of numpy arrays)
        detail_coeffs_list = coeffs[1:] # cD_n, cD_n-1, ..., cD_1

        # --- MAD Calculation ---
        # Usually, the threshold is estimated from the finest level of detail coefficients (cD_1), which is coeffs[-1]
        first_level_details = coeffs[-1]

        # Check if the detail coefficients array is empty or too small
        if first_level_details is None or first_level_details.size == 0:
            logger.warning("First level detail coefficients are empty. Cannot estimate noise variance. Returning original signal.")
            return signal.reindex(original_index)

        # Calculate MAD using statsmodels.robust.mad for numpy arrays
        # This is more robust than pandas' mad and works directly on numpy arrays.
        # c=1 ensures it computes the Median Absolute Deviation from the median.
        try:
            # Calculate MAD, handling potential all-zero arrays which cause issues
            median_val = np.median(first_level_details)
            mad_value = sm_mad(first_level_details, c=1) # Use statsmodels MAD

            if np.isnan(mad_value) or mad_value < 1e-9: # Check for NaN or effectively zero MAD
                 logger.warning(f"MAD calculation resulted in NaN or near-zero ({mad_value}). Check for constant signals or insufficient variation. Using small default sigma.")
                 sigma = 1e-6 # Assign a small default sigma to avoid division by zero or instability
            else:
                 sigma = mad_value / 0.6745 # Correct scaling factor for Gaussian noise
        except ValueError as ve:
            logger.error(f"Error calculating MAD: {ve}. Check input data. Returning original signal.")
            # Print traceback for debugging
            import traceback
            logger.error(traceback.format_exc())
            return signal.reindex(original_index)


        # --- Thresholding ---
        # Universal threshold
        threshold = sigma * np.sqrt(2 * np.log(len(signal_clean)))

        # Threshold the detail coefficients
        new_coeffs = [coeffs[0]] # Keep approximation coefficients
        for detail_coeff_array in detail_coeffs_list:
             new_coeffs.append(pywt.threshold(detail_coeff_array, value=threshold, mode='soft'))

        # --- Reconstruction ---
        # Reconstruct the signal using the same length logic as pywt
        try:
            denoised_signal_values = pywt.waverec(new_coeffs, wavelet)
        except ValueError as vr:
             logger.error(f"Error during wavelet reconstruction: {vr}. Check coefficients. Returning original signal.")
             # Print traceback for debugging
             import traceback
             logger.error(traceback.format_exc())
             return signal.reindex(original_index)


        # --- Aligning Length and Index ---
        # Ensure the reconstructed signal length matches the *cleaned* signal length
        # Use the length that pywt expects for reconstruction consistency (often signal_clean length)
        target_length = len(signal_clean)
        if len(denoised_signal_values) != target_length:
            logger.warning(f"Length mismatch after wavelet reconstruction: expected {target_length}, got {len(denoised_signal_values)}. Padding/Truncating.")
            # Pad with zeros or truncate to match the target length
            if len(denoised_signal_values) > target_length:
                denoised_signal_values = denoised_signal_values[:target_length]
            else:
                padding = target_length - len(denoised_signal_values)
                denoised_signal_values = np.pad(denoised_signal_values, (0, padding), 'constant')

        # Create a Pandas Series with the index of the *cleaned* signal
        denoised_signal = pd.Series(denoised_signal_values, index=signal_clean.index)

        # Reindex to match the original signal's index, filling missing values (NaNs that were dropped) with NaN
        denoised_signal = denoised_signal.reindex(original_index)

        # Final check for all NaNs after reindexing (shouldn't happen if handled above)
        if denoised_signal.isnull().all():
             logger.warning("Denoised signal is all NaN after reindexing. Returning original signal.")
             return signal.reindex(original_index)


        return denoised_signal

    except Exception as e:
        logger.error(f"Wavelet denoising failed unexpectedly: {e}")
        # Print traceback for debugging
        import traceback
        logger.error(traceback.format_exc())
        # Return the original signal reindexed if denoising fails
        return signal.reindex(original_index)

def moving_average_smooth(data_series, window_size=5):
    """使用移动平均进行平滑。"""
    if window_size <= 1:
        return data_series
    return data_series.rolling(window=window_size, center=True, min_periods=1).mean()

def savgol_smooth(data_series, window_size=5, poly_order=2):
    """使用 Savitzky-Golay 滤波器进行平滑。"""
    if window_size <= poly_order or window_size % 2 == 0:
        logger.warning(f"Adjusting Sav-Gol window size from {window_size} to {max(poly_order + 1, window_size + (1 if window_size % 2 == 0 else 0))}")
        window_size = max(poly_order + 1, window_size + (1 if window_size % 2 == 0 else 0))
    try:
        smoothed = savgol_filter(data_series.dropna(), window_size, poly_order)
        return pd.Series(smoothed, index=data_series.dropna().index).reindex(data_series.index)
    except Exception as e:
        logger.error(f"Savitzky-Golay smoothing failed: {e}")
        return data_series

def preprocess_signals(data_df, config):
    """对 DataFrame 中的信号列进行预处理。

    Args:
        data_df (pd.DataFrame): 包含原始信号数据的 DataFrame。
                                 需要知道哪些列是信号列。
        config (dict): 包含预处理参数的配置字典。
                       例如: config['preprocessing']['denoising_method'],
                             config['preprocessing']['smoothing_window']

    Returns:
        pd.DataFrame: 包含预处理后信号的 DataFrame。
    """
    processed_df = data_df.copy()
    preprocessing_cfg = config.get('preprocessing', {})
    denoising_method = preprocessing_cfg.get('denoising_method', 'wavelet')
    smoothing_window = preprocessing_cfg.get('smoothing_window', 5)
    outlier_method = preprocessing_cfg.get('outlier_method', 'zscore')  # 默认使用zscore
    zscore_threshold = preprocessing_cfg.get('zscore_threshold', 3.0)   # zscore阈值
    iqr_factor = preprocessing_cfg.get('iqr_factor', 1.5)               # IQR阈值因子
    wavelet_cfg = config.get('feature_extraction', {}).get('wavelet_packet', {})
    wavelet_type = wavelet_cfg.get('wavelet_type', 'db4') # 假设小波类型配置在此
    wavelet_level = wavelet_cfg.get('level', 1) # 或者专门为去噪设置级别

    # --- 确定要处理的信号列 --- S
    # !!! 重要: 您需要定义哪些列是需要预处理的信号列 !!!
    # 这可能基于列名约定、配置文件或数据集本身的信息
    # 示例：假设所有非 'time' 和 非 'wear'/'RUL' 的列都是信号
    exclude_cols = ['time', 'wear', 'RUL', 'experiment_id', 'measurement_id'] 
    signal_columns = [col for col in processed_df.columns if col not in exclude_cols] # 示例排除列
    logger.info(f"Preprocessing signal columns: {signal_columns}")
    if not signal_columns:
         logger.warning("No signal columns identified for preprocessing.")
         return processed_df # Or raise error

    for col in signal_columns:
        logger.debug(f"Processing column: {col}")
        signal = processed_df[col].astype(float) # 确保是数值类型

        # 1. 异常值处理 (可选，但推荐)
        if outlier_method == 'zscore':
            logger.debug(f"Applying Z-score outlier removal to {col} with threshold {zscore_threshold}")
            signal = remove_outliers_zscore(signal, threshold=zscore_threshold)
        elif outlier_method == 'iqr':
            logger.debug(f"Applying IQR outlier removal to {col} with factor {iqr_factor}")
            signal = remove_outliers_iqr(signal, factor=iqr_factor)
        else:
            logger.debug(f"No outlier removal method specified for {col}. Using default Z-score method.")
            signal = remove_outliers_zscore(signal)  # 默认使用Z-score
            
        # 处理因异常值移除产生的 NaN (例如，使用插值或保留)
        signal = signal.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

        # 2. 去噪
        if denoising_method == 'wavelet':
            logger.debug(f"Applying wavelet denoising to {col}")
            signal = wavelet_denoise(signal, wavelet=wavelet_type, level=wavelet_level)
        elif denoising_method == 'moving_average': # 示例：如果配置了移动平均去噪
             logger.debug(f"Applying moving average (as denoising) to {col}")
             signal = moving_average_smooth(signal, window_size=smoothing_window) # 使用平滑窗口作为去噪窗口?
        elif denoising_method:
            logger.warning(f"Unsupported denoising method '{denoising_method}' for {col}. Skipping denoising.")

        # 3. 平滑
        if smoothing_window > 1:
             logger.debug(f"Applying Savitzky-Golay smoothing to {col} with window {smoothing_window}")
             # Savitzky-Golay 通常比移动平均效果更好，且保留信号特征
             signal = savgol_smooth(signal, window_size=smoothing_window, poly_order=2) # poly_order 可配置
            # signal = moving_average_smooth(signal, window_size=smoothing_window)

        processed_df[col] = signal

    logger.info("Signal preprocessing complete.")
    return processed_df

# # --- 示例用法 ---
# if __name__ == '__main__':
#     import yaml
#     from loader import load_phm2010_data # 假设 loader.py 在同级目录
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#     # 加载配置
#     try:
#         with open('../../config/config.yaml', 'r') as f:
#             config = yaml.safe_load(f)
#     except Exception as e:
#         print(f"Error loading config: {e}")
#         config = {}

#     # 加载数据 (需要先运行 loader 示例或确保数据已加载)
#     raw_data_directory = config.get('data', {}).get('dataset_path', '../../data/raw/PHM_2010/')
#     all_data = load_phm2010_data(raw_data_directory)

#     if all_data:
#         # 选择一个实验进行预处理
#         exp_key = list(all_data.keys())[0]
#         print(f"\nPreprocessing data for experiment: {exp_key}")
#         sample_df = all_data[exp_key]

#         # !!! 重要: 模拟信号列名 (需要替换为真实列名) !!!
#         # 假设前 7 列是信号
#         num_cols = sample_df.shape[1]
#         if num_cols > 7:
#              sample_df.columns = [f'sensor_{i}' for i in range(num_cols-1)] + ['wear']
#         else:
#              sample_df.columns = [f'sensor_{i}' for i in range(num_cols)]
#         print("Original data head:")
#         print(sample_df.head())

#         processed_data = preprocess_signals(sample_df, config)

#         print("\nProcessed data head:")
#         print(processed_data.head())

#         # (可选) 绘制对比图
#         # import matplotlib.pyplot as plt
#         # signal_col_to_plot = 'sensor_0' # 选择一列绘制
#         # plt.figure(figsize=(12, 6))
#         # plt.plot(sample_df.index, sample_df[signal_col_to_plot], label='Original', alpha=0.7)
#         # plt.plot(processed_data.index, processed_data[signal_col_to_plot], label='Processed', linewidth=2)
#         # plt.title(f'Preprocessing Comparison for {signal_col_to_plot}')
#         # plt.legend()
#         # plt.show()
#     else:
#         print("Cannot run preprocessing example, data loading failed.")
