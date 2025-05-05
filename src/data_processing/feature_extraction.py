import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.fft import fft, fftfreq
import pywt
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# --- Time Domain Features ---

def calculate_rms(series):
    return np.sqrt(np.mean(series**2))

def calculate_peak(series):
    return np.max(np.abs(series))

def calculate_crest_factor(series):
    peak = calculate_peak(series)
    rms = calculate_rms(series)
    return peak / rms if rms != 0 else np.nan

def calculate_peak_to_peak(series):
    """计算峰峰值 (最大值与最小值之差)"""
    return series.max() - series.min()

def calculate_form_factor(series):
    """计算波形因子 (RMS / 平均绝对值)"""
    mean_abs = np.mean(np.abs(series))
    rms = calculate_rms(series)
    return rms / mean_abs if mean_abs != 0 else np.nan

def calculate_impulse_factor(series):
    """计算脉冲因子 (峰值 / 平均绝对值)"""
    mean_abs = np.mean(np.abs(series))
    peak = calculate_peak(series)
    return peak / mean_abs if mean_abs != 0 else np.nan

def calculate_clearance_factor(series):
    """计算裕度因子 (峰值 / 均方根平方)"""
    mean_square_root = np.mean(np.sqrt(np.abs(series)))
    peak = calculate_peak(series)
    return peak / mean_square_root if mean_square_root != 0 else np.nan

def calculate_shape_factor(series):
    """计算波形形状因子 (均方根 / 平均绝对值)"""
    mean_abs = np.mean(np.abs(series))
    rms = calculate_rms(series)
    return rms / mean_abs if mean_abs != 0 else np.nan

# --- Frequency Domain Features ---

def calculate_fft_features(series, sampling_rate=50000): # PHM 2010 is 50kHz
    """计算频域特征，包括主频幅值、能量比、谱熵、谱质心、谱峭度和频带能量比"""
    N = len(series)
    if N <= 1:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    yf = fft(series.to_numpy())
    xf = fftfreq(N, 1 / sampling_rate)

    # 仅使用正频率
    mask = xf >= 0
    xf = xf[mask]
    yf_magnitude = np.abs(yf[mask]) * 2 / N # 归一化幅值

    if len(xf) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # 1. 主频幅值
    main_freq_idx = np.argmax(yf_magnitude)
    main_freq_amplitude = yf_magnitude[main_freq_idx]

    # 2. 能量比
    total_energy = np.sum(yf_magnitude**2)
    dominant_energy = yf_magnitude[main_freq_idx]**2
    energy_ratio = dominant_energy / total_energy if total_energy != 0 else np.nan

    # 3. 谱熵
    # 归一化功率谱以获得"概率分布"
    norm_ps = yf_magnitude**2 / total_energy if total_energy != 0 else np.zeros_like(yf_magnitude)
    # 计算谱熵
    spectral_entropy = -np.sum(norm_ps * np.log2(norm_ps + 1e-12))  # 避免log(0)

    # 4. 谱质心
    spectral_centroid = np.sum(xf * norm_ps) / np.sum(norm_ps) if np.sum(norm_ps) != 0 else np.nan

    # 5. 谱峭度
    # 从功率谱计算峭度
    spectral_variance = np.sum(((xf - spectral_centroid) ** 2) * norm_ps) if np.sum(norm_ps) != 0 else np.nan
    spectral_kurtosis = np.sum(((xf - spectral_centroid) ** 4) * norm_ps) / (spectral_variance ** 2) if spectral_variance != 0 else np.nan

    # 6. 频带能量比
    # 将频率范围分为低、中、高三个频带
    max_freq = xf[-1]
    # 低频段: 0-33%
    low_band_mask = (xf >= 0) & (xf < max_freq * 0.33)
    # 中频段: 33%-66%
    mid_band_mask = (xf >= max_freq * 0.33) & (xf < max_freq * 0.66)
    # 高频段: 66%-100%
    high_band_mask = (xf >= max_freq * 0.66)

    # 计算每个频带的能量
    low_band_energy = np.sum(yf_magnitude[low_band_mask]**2)
    mid_band_energy = np.sum(yf_magnitude[mid_band_mask]**2)
    high_band_energy = np.sum(yf_magnitude[high_band_mask]**2)

    # 计算每个频带占总能量的比例
    freq_band_energy_ratio_low = low_band_energy / total_energy if total_energy != 0 else np.nan
    freq_band_energy_ratio_mid = mid_band_energy / total_energy if total_energy != 0 else np.nan
    freq_band_energy_ratio_high = high_band_energy / total_energy if total_energy != 0 else np.nan

    return (main_freq_amplitude, energy_ratio, spectral_entropy, spectral_centroid, 
            spectral_kurtosis, freq_band_energy_ratio_low, freq_band_energy_ratio_mid, 
            freq_band_energy_ratio_high)

# --- Time-Frequency Domain Features (Wavelet Packet Decomposition) ---

def calculate_wpd_energy_entropy(series, wavelet='db4', level=4):
    """Calculates energy entropy for each terminal node of WPD."""
    signal = series.dropna().to_numpy()
    min_len = 2**level
    num_bands = min_len # Number of expected output features/bands

    # Prepare default NaN return dictionary
    feature_names = [f'wpd_L{level}_E{i}' for i in range(num_bands)]
    nan_result = dict(zip(feature_names, [np.nan] * num_bands))

    if len(signal) < min_len:
        # Log warning only once per level/wavelet combo to avoid flood? Might need better logging strategy.
        # For now, log every time.
        logger.warning(f"Signal length ({len(signal)}) is too short for WPD level {level} (requires {min_len}). Skipping WPD for this segment.")
        return nan_result

    try:
        wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='per')
        # Check max level just in case, although len check should prevent this specific error
        max_level = wp.maxlevel
        if level > max_level:
             logger.warning(f"Requested WPD level {level} exceeds max possible level {max_level} for signal length {len(signal)}. Skipping WPD.")
             return nan_result

        nodes = wp.get_level(level, order='natural') # Get nodes at the specified level
        energies = [np.sum(node.data**2) for node in nodes]
        total_energy = np.sum(energies)

        if total_energy == 0:
             # Return NaN or zeros for each sub-band if total energy is zero
             logger.debug(f"Total energy for WPD level {level} is zero. Returning NaNs.") # Use debug level
             return nan_result

        # Normalize energies to get probabilities
        probabilities = np.array(energies) / total_energy
        # Calculate entropy for each node (sub-band)
        # Entropy = -sum(p * log2(p)) where p is normalized coefficient energy within the sub-band
        # For simplicity here, we return the *energy* of each band, not entropy yet.
        # To calculate entropy per band, we need coefficients within each band.

        # Simpler approach: Return the energy of each sub-band directly
        # Or calculate overall entropy based on band energies
        # Let's return band energies as features first, as entropy calculation is more involved.
        # Later, we can implement entropy: -np.sum(p * np.log2(p + 1e-12)) for p = energies / total_energy

        # Use normalized energy as feature per band
        feature_values = probabilities
        if len(feature_values) < num_bands:
             # Pad with NaNs if fewer nodes were generated
             logger.warning(f"WPD returned {len(feature_values)} nodes, expected {num_bands}. Padding with NaNs.")
             feature_values = np.pad(feature_values, (0, num_bands - len(feature_values)), constant_values=np.nan)
        elif len(feature_values) > num_bands:
             logger.warning(f"WPD returned {len(feature_values)} nodes, expected {num_bands}. Truncating.")
             feature_values = feature_values[:num_bands]

        return dict(zip(feature_names, feature_values))

    except Exception as e:
        logger.error(f"WPD failed unexpectedly for level {level}, wavelet {wavelet} even after length check: {e}", exc_info=True)
        return nan_result


# --- Main Extraction Function ---

def extract_features_for_measurement(measurement_df, signal_cols, config, measurement_id):
    """Extracts features from all signal columns for a single measurement_id."""
    features = {}
    fe_config = config.get('feature_extraction', {})
    time_features_list = fe_config.get('time_domain_features', [])
    freq_features_list = fe_config.get('frequency_domain_features', [])
    wpd_config = fe_config.get('wavelet_packet', {})
    wpd_level = wpd_config.get('level', 4)
    # 从配置获取小波类型
    wpd_wavelet = wpd_config.get('wavelet_type', 'db4')

    sampling_rate = 50000 # Assuming PHM2010 default

    for col in signal_cols:
        series = measurement_df[col]
        if series.empty or series.isnull().all(): # Check if series is empty OR all NaN AFTER assignment
            logger.warning(f"Signal column {col} is empty or all NaNs for this measurement. Skipping feature extraction for it.")
            # Add NaNs for all features of this column
            if 'mean' in time_features_list: features[f'{col}_mean'] = np.nan
            if 'std' in time_features_list: features[f'{col}_std'] = np.nan
            if 'rms' in time_features_list: features[f'{col}_rms'] = np.nan
            if 'skewness' in time_features_list: features[f'{col}_skewness'] = np.nan
            if 'kurtosis' in time_features_list: features[f'{col}_kurtosis'] = np.nan
            if 'peak' in time_features_list: features[f'{col}_peak'] = np.nan
            if 'crest_factor' in time_features_list: features[f'{col}_crest_factor'] = np.nan
            if 'peak_to_peak' in time_features_list: features[f'{col}_peak_to_peak'] = np.nan
            if 'form_factor' in time_features_list: features[f'{col}_form_factor'] = np.nan
            if 'impulse_factor' in time_features_list: features[f'{col}_impulse_factor'] = np.nan
            if 'clearance_factor' in time_features_list: features[f'{col}_clearance_factor'] = np.nan
            if 'shape_factor' in time_features_list: features[f'{col}_shape_factor'] = np.nan
            
            # 频域特征
            if 'main_frequency_amplitude' in freq_features_list: features[f'{col}_main_freq_amp'] = np.nan
            if 'energy_ratio' in freq_features_list: features[f'{col}_energy_ratio'] = np.nan
            if 'spectral_entropy' in freq_features_list: features[f'{col}_spectral_entropy'] = np.nan
            if 'spectral_centroid' in freq_features_list: features[f'{col}_spectral_centroid'] = np.nan
            if 'spectral_kurtosis' in freq_features_list: features[f'{col}_spectral_kurtosis'] = np.nan
            if 'freq_band_energy_ratio_low' in freq_features_list: features[f'{col}_freq_band_energy_ratio_low'] = np.nan
            if 'freq_band_energy_ratio_mid' in freq_features_list: features[f'{col}_freq_band_energy_ratio_mid'] = np.nan
            if 'freq_band_energy_ratio_high' in freq_features_list: features[f'{col}_freq_band_energy_ratio_high'] = np.nan
            
            # WPD特征
            num_bands = 2**wpd_level
            for i in range(num_bands):
                features[f'{col}_wpd_L{wpd_level}_E{i}'] = np.nan
            continue

        # 时域特征提取
        if 'mean' in time_features_list: features[f'{col}_mean'] = series.mean()
        if 'std' in time_features_list: features[f'{col}_std'] = series.std()
        if 'rms' in time_features_list: features[f'{col}_rms'] = calculate_rms(series)
        if 'skewness' in time_features_list: features[f'{col}_skewness'] = sp_stats.skew(series)
        if 'kurtosis' in time_features_list: features[f'{col}_kurtosis'] = sp_stats.kurtosis(series)
        if 'peak' in time_features_list: features[f'{col}_peak'] = calculate_peak(series)
        if 'crest_factor' in time_features_list: features[f'{col}_crest_factor'] = calculate_crest_factor(series)
        if 'peak_to_peak' in time_features_list: features[f'{col}_peak_to_peak'] = calculate_peak_to_peak(series)
        if 'form_factor' in time_features_list: features[f'{col}_form_factor'] = calculate_form_factor(series)
        if 'impulse_factor' in time_features_list: features[f'{col}_impulse_factor'] = calculate_impulse_factor(series)
        if 'clearance_factor' in time_features_list: features[f'{col}_clearance_factor'] = calculate_clearance_factor(series)
        if 'shape_factor' in time_features_list: features[f'{col}_shape_factor'] = calculate_shape_factor(series)

        # 频域特征提取
        if freq_features_list:
            (main_amp, energy_ratio, spectral_entropy, spectral_centroid, 
             spectral_kurtosis, freq_band_energy_ratio_low, freq_band_energy_ratio_mid, 
             freq_band_energy_ratio_high) = calculate_fft_features(series, sampling_rate)
            
            if 'main_frequency_amplitude' in freq_features_list:
                features[f'{col}_main_freq_amp'] = main_amp
            if 'energy_ratio' in freq_features_list:
                features[f'{col}_energy_ratio'] = energy_ratio
            if 'spectral_entropy' in freq_features_list:
                features[f'{col}_spectral_entropy'] = spectral_entropy
            if 'spectral_centroid' in freq_features_list:
                features[f'{col}_spectral_centroid'] = spectral_centroid
            if 'spectral_kurtosis' in freq_features_list:
                features[f'{col}_spectral_kurtosis'] = spectral_kurtosis
            if 'freq_band_energy_ratio_low' in freq_features_list:
                features[f'{col}_freq_band_energy_ratio_low'] = freq_band_energy_ratio_low
            if 'freq_band_energy_ratio_mid' in freq_features_list:
                features[f'{col}_freq_band_energy_ratio_mid'] = freq_band_energy_ratio_mid
            if 'freq_band_energy_ratio_high' in freq_features_list:
                features[f'{col}_freq_band_energy_ratio_high'] = freq_band_energy_ratio_high

        # Time-Frequency Domain (WPD)
        if wpd_level > 0:
            # ---> 添加日志 <---
            non_nan_count = series.notna().sum()
            min_len_wpd = 2**wpd_level
            if non_nan_count < min_len_wpd:
                 logger.warning(f"Preparing WPD for col '{col}', measurement_id {measurement_id}: Series has only {non_nan_count} non-NaN values before internal dropna (requires {min_len_wpd}). Length: {len(series)}")
            # ---> 日志结束 <--- 
            # calculate_wpd_energy_entropy already handles dropna internally
            wpd_features = calculate_wpd_energy_entropy(series, wavelet=wpd_wavelet, level=wpd_level)
            for fname, fval in wpd_features.items():
                 features[f'{col}_{fname}'] = fval # Add prefix

    return features

def extract_features_from_data(processed_data_dict, config):
    """Extracts features from the preprocessed data dictionary.

    Args:
        processed_data_dict (dict): Dictionary where keys are experiment names
                                   and values are preprocessed DataFrames
                                   (output of preprocess_signals).
                                   Each DataFrame must contain 'measurement_id',
                                   'experiment_id', 'wear', and signal columns.
        config (dict): Configuration dictionary.

    Returns:
        pd.DataFrame: A DataFrame containing extracted features for each measurement,
                      indexed by ('experiment_id', 'measurement_id'), plus a 'wear' column.
                      Returns None if errors occur.
    """
    all_features_list = []
    required_cols = ['measurement_id', 'experiment_id', 'wear'] # Need these cols

    logger.info("Starting feature extraction process...")

    for exp_name, data_df in processed_data_dict.items():
        logger.info(f"Extracting features for experiment: {exp_name}")

        if not all(col in data_df.columns for col in required_cols):
            logger.error(f"Experiment {exp_name} DataFrame missing required columns ({required_cols}). Found: {data_df.columns}. Skipping.")
            continue

        # Identify signal columns (assuming all others are IDs or target)
        signal_columns = [col for col in data_df.columns if col not in required_cols]
        if not signal_columns:
             logger.warning(f"No signal columns found for feature extraction in {exp_name}. Skipping.")
             continue
        logger.info(f"Using signal columns for {exp_name}: {signal_columns}")

        # Group by measurement_id and apply feature extraction
        grouped = data_df.groupby('measurement_id')
        measurement_features = []

        total_measurements = len(grouped)
        processed_count = 0

        for measurement_id, measurement_group in grouped:
            try:
                exp_id = measurement_group['experiment_id'].iloc[0] # Should be same for group
                wear_value = measurement_group['wear'].iloc[0] # Assuming wear is constant per measurement

                # Extract features for this specific measurement's signals
                features = extract_features_for_measurement(measurement_group, signal_columns, config, measurement_id)

                # Add identifiers and target
                features['measurement_id'] = measurement_id
                features['experiment_id'] = exp_id
                features['wear'] = wear_value

                measurement_features.append(features)
                processed_count += 1
                if processed_count % 50 == 0: # Log progress periodically
                     logger.info(f"Processed {processed_count}/{total_measurements} measurements for {exp_name}...")

            except Exception as e:
                logger.error(f"Error extracting features for measurement {measurement_id} in {exp_name}: {e}", exc_info=True)
                # Optionally add a row with NaNs or skip

        if measurement_features:
            all_features_list.extend(measurement_features)
        logger.info(f"Finished feature extraction for {exp_name}. Extracted features for {len(measurement_features)} measurements.")


    if not all_features_list:
        logger.error("No features were extracted from any experiment.")
        return None

    # Combine all features into a single DataFrame
    features_df = pd.DataFrame(all_features_list)

    # Set multi-index
    try:
        features_df = features_df.set_index(['experiment_id', 'measurement_id'])
    except KeyError:
         logger.error("Could not set multi-index. 'experiment_id' or 'measurement_id' missing from extracted features.")
         # Fallback to default index if needed
         pass

    # Separate target variable
    if 'wear' in features_df.columns:
        target = features_df.pop('wear')
    else:
        logger.error("'wear' column not found in the final features DataFrame!")
        target = None # Or handle differently

    logger.info(f"Feature extraction complete. Final feature matrix shape: {features_df.shape}")
    # Return features DataFrame and target Series separately
    return features_df, target


# # --- Example Usage ---
# if __name__ == '__main__':
#     import yaml
#     from loader import load_phm2010_data
#     from preprocessing import preprocess_signals # Assume preprocessing is done first
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#     # 1. Load Config
#     config_path = '../../config/config.yaml'
#     try:
#         with open(config_path, 'r') as f:
#             config = yaml.safe_load(f)
#     except Exception as e:
#         print(f"Error loading config: {e}")
#         config = {}

#     # 2. Load Data (using the updated loader)
#     script_dir = os.path.dirname(__file__)
#     project_root = os.path.abspath(os.path.join(script_dir, '../../'))
#     root_data_dir = config.get('data', {}).get('dataset_path', 'data/raw/PHM_2010/')
#     root_data_dir_abs = os.path.abspath(os.path.join(project_root, root_data_dir))
#     raw_data_dict = load_phm2010_data(root_data_dir_abs)

#     if not raw_data_dict:
#          print("Failed to load raw data. Cannot proceed with feature extraction example.")
#          exit()

#     # 3. Preprocess Data (apply to each experiment's DataFrame)
#     processed_data = {}
#     # Let's process only one experiment for the example to save time
#     exp_to_process = 'c1' # Or choose another like 'c4', 'c6'
#     if exp_to_process in raw_data_dict:
#         print(f"\nPreprocessing {exp_to_process} data...")
#         processed_data[exp_to_process] = preprocess_signals(raw_data_dict[exp_to_process], config)
#         print(f"Finished preprocessing {exp_to_process}.")
#     else:
#         print(f"Experiment {exp_to_process} not found in loaded data.")
#         exit()

#     # Handle case where preprocessing failed or returned None
#     if not processed_data or processed_data[exp_to_process] is None:
#          print(f"Preprocessing failed for {exp_to_process}. Cannot proceed.")
#          exit()


#     # 4. Extract Features
#     print(f"\nExtracting features from preprocessed {exp_to_process} data...")
#     features_df, wear_series = extract_features_from_data(processed_data, config)

#     if features_df is not None and wear_series is not None:
#         print("\nFeature extraction successful!")
#         print("Features DataFrame shape:", features_df.shape)
#         print("Features DataFrame head:")
#         print(features_df.head())
#         print("\nWear Series shape:", wear_series.shape)
#         print("Wear Series head:")
#         print(wear_series.head())
#     else:
#         print("\nFeature extraction failed.")


