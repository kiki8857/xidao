import pandas as pd
import numpy as np
import yaml
import logging
import os
import argparse
import joblib
from datetime import datetime
import sys
import glob # To find processed files

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

# Import necessary functions from src
try:
    from src.data_processing.feature_extraction import extract_features_from_data
    from src.data_processing.feature_selection import select_features
    from src.data_processing.dimensionality_reduction import apply_pca
    from src.utils.logger import setup_logger
except ImportError as e:
    raise ImportError(f"Could not import project modules. Ensure PYTHONPATH is set correctly or run from root: {e}")

def load_processed_data(processed_data_dir, logger):
    """Loads processed CSV files from the specified directory."""
    processed_files = glob.glob(os.path.join(processed_data_dir, '*_processed.csv'))
    if not processed_files:
        logger.error(f"No *_processed.csv files found in {processed_data_dir}. Run preprocess_data.py first.")
        return None

    processed_data_dict = {}
    for f_path in processed_files:
        try:
            exp_name = os.path.basename(f_path).replace('_processed.csv', '')
            logger.info(f"Loading processed data for experiment: {exp_name} from {f_path}")
            processed_data_dict[exp_name] = pd.read_csv(f_path)
        except Exception as e:
            logger.error(f"Error loading processed file {f_path}: {e}", exc_info=True)
            # Decide whether to continue or abort if one file fails
            # return None # Abort
            continue # Skip this file
            
    if not processed_data_dict:
         logger.error("Failed to load any processed data.")
         return None
         
    logger.info(f"Successfully loaded processed data for experiments: {list(processed_data_dict.keys())}")
    return processed_data_dict

def engineer_features(config_path):
    """Loads processed data, extracts/selects features, applies PCA, and saves results."""

    # 1. Load configuration
    print(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        return

    # --- Create results/log directory for this stage ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.abspath(os.path.join(project_root, 'logs', f'feature_engineering_{timestamp}'))
    # Also create a dir to save PCA/scaler models for this run
    pca_save_dir = os.path.abspath(os.path.join(project_root, 'results', f'feature_engineering_{timestamp}'))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(pca_save_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'feature_engineering.log')

    # --- Configure Logger ---
    log_level_str = config.get('logging', {}).get('log_level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger = setup_logger('FeatureEngLogger', log_level, log_file_path)
    logger.info(f"Logger setup complete. Log level: {log_level_str}. Log file: {log_file_path}")
    logger.info(f"Using configuration from: {config_path}")
    logger.info(f"PCA/Scaler models for this run will be saved in: {pca_save_dir}")

    # --- Get Processed Data Load Path and Final Data Save Path ---
    processed_data_dir_rel = config.get('data', {}).get('processed_data_dir', 'data/processed')
    processed_data_dir = os.path.abspath(os.path.join(project_root, processed_data_dir_rel))
    final_features_save_path = os.path.join(processed_data_dir, 'final_features.csv')
    target_save_path = os.path.join(processed_data_dir, 'target_wear.csv')
    logger.info(f"Loading processed data from: {processed_data_dir}")
    logger.info(f"Final features will be saved to: {final_features_save_path}")
    logger.info(f"Target variable will be saved to: {target_save_path}")

    # 2. Load Processed Data
    logger.info("--- Step 1: Loading Processed Data ---")
    processed_data_dict = load_processed_data(processed_data_dir, logger)
    if not processed_data_dict:
        logger.error("Failed to load processed data. Aborting feature engineering.")
        return

    # 3. Feature Extraction
    logger.info("--- Step 2: Extracting Features ---")
    features_df, target_series = extract_features_from_data(processed_data_dict, config)
    del processed_data_dict # Release memory
    if features_df is None or target_series is None:
        logger.error("Feature extraction failed. Aborting.")
        return
    logger.info(f"Initial features extracted. Shape: {features_df.shape}. Target shape: {target_series.shape}")

    # 4. Feature Selection
    logger.info("--- Step 3: Selecting Features ---")
    selected_features_df = select_features(features_df, target_series, config)
    if selected_features_df is None or selected_features_df.empty:
        logger.error("Feature selection resulted in empty or None DataFrame. Aborting.")
        return
    # Ensure target series aligns with selected features' index
    target_series = target_series.loc[selected_features_df.index]
    logger.info(f"Features selected. Shape: {selected_features_df.shape}")
    del features_df # Release memory

    # 5. Dimensionality Reduction (PCA)
    logger.info("--- Step 4: Applying PCA ---")
    # Pass the dedicated save path for PCA/scaler models for this run
    pca_result = apply_pca(selected_features_df, config, save_path=pca_save_dir)
    if pca_result is None:
        logger.error("PCA failed. Aborting.")
        return
    # apply_pca returns tuple (pca_df, scaler, pca_model) when save_path is provided
    final_features_df, scaler_model, pca_model = pca_result 
    logger.info(f"PCA applied. Final features shape: {final_features_df.shape}")
    logger.info(f"Scaler and PCA models saved to {pca_save_dir}")
    del selected_features_df # Release memory

    # 6. Save Final Features and Target
    logger.info("--- Step 5: Saving Final Features and Target ---")
    try:
        # Save features (including index: experiment_id, measurement_id)
        final_features_df.to_csv(final_features_save_path, index=True)
        # Save target (including index)
        target_series.to_csv(target_save_path, index=True, header=True) # Save header for series name
        logger.info(f"Final features saved to {final_features_save_path}")
        logger.info(f"Target variable saved to {target_save_path}")
    except Exception as e:
        logger.error(f"Error saving final features or target: {e}", exc_info=True)
        return

    logger.info(f"--- Feature Engineering Stage Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Engineering for PHM 2010 Data")
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

    engineer_features(config_file_path) 