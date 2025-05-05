import pandas as pd
import numpy as np
import yaml
import logging
import os
import argparse
from datetime import datetime
import sys

# --- Add project root to sys.path ---
# Assuming this script is in 'scripts/' directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

# Import necessary functions from src
try:
    from src.data_processing.loader import load_phm2010_data
    from src.data_processing.preprocessing import preprocess_signals
    from src.utils.logger import setup_logger
except ImportError as e:
    raise ImportError(f"Could not import project modules. Ensure PYTHONPATH is set correctly or run from root: {e}")

def preprocess_data(config_path):
    """Loads raw data, preprocesses signals, and saves processed data."""

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
    # Create a specific log dir for preprocessing stage
    log_dir = os.path.abspath(os.path.join(project_root, 'logs', f'preprocess_{timestamp}'))
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'preprocess.log')

    # --- Configure Logger ---
    log_level_str = config.get('logging', {}).get('log_level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger = setup_logger('PreprocessLogger', log_level, log_file_path)
    logger.info(f"Logger setup complete. Log level: {log_level_str}. Log file: {log_file_path}")
    logger.info(f"Using configuration from: {config_path}")

    # --- Get Processed Data Save Path ---
    processed_data_dir_rel = config.get('data', {}).get('processed_data_dir', 'data/processed')
    processed_data_dir = os.path.abspath(os.path.join(project_root, processed_data_dir_rel))
    os.makedirs(processed_data_dir, exist_ok=True)
    logger.info(f"Processed data will be saved in: {processed_data_dir}")

    # 2. Load Raw Data
    logger.info("--- Step 1: Loading Raw Data ---")
    data_cfg = config.get('data', {})
    raw_data_dir_rel = data_cfg.get('dataset_path', 'data/raw/PHM_2010/')
    if not os.path.isabs(raw_data_dir_rel):
        raw_data_dir = os.path.abspath(os.path.join(project_root, raw_data_dir_rel))
    else:
        raw_data_dir = raw_data_dir_rel

    raw_data_dict = load_phm2010_data(raw_data_dir)
    if not raw_data_dict:
        logger.error("Failed to load raw data. Aborting.")
        return
    logger.info(f"Loaded raw data for experiments: {list(raw_data_dict.keys())}")

    # 3. Preprocess Data and Save
    logger.info("--- Step 2: Preprocessing Data and Saving ---")
    processed_count = 0
    failed_count = 0
    for exp_name, df in raw_data_dict.items():
        logger.info(f"Preprocessing data for experiment: {exp_name}...")
        try:
            processed_df = preprocess_signals(df, config)
            if processed_df is not None and not processed_df.empty:
                save_filename = f"{exp_name}_processed.csv"
                save_path = os.path.join(processed_data_dir, save_filename)
                processed_df.to_csv(save_path, index=False) # Save without index if not meaningful
                logger.info(f"Successfully processed and saved {save_filename} to {processed_data_dir}")
                processed_count += 1
            else:
                logger.warning(f"Preprocessing returned empty or None DataFrame for experiment {exp_name}. Skipping save.")
                failed_count += 1
        except Exception as e:
            logger.error(f"Error during preprocessing for experiment {exp_name}: {e}", exc_info=True)
            failed_count += 1
        # Optional: Clear memory for the raw dataframe of this experiment if needed
        # raw_data_dict[exp_name] = None 
        # del df

    logger.info(f"--- Preprocessing Stage Finished ---")
    logger.info(f"Successfully processed and saved data for {processed_count} experiments.")
    if failed_count > 0:
        logger.warning(f"Preprocessing failed or was skipped for {failed_count} experiments.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PHM 2010 Data")
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to the configuration file (default: config/config.yaml)')
    args = parser.parse_args()

    # Ensure the config path is absolute or relative to the project root
    if not os.path.isabs(args.config):
        config_file_path = os.path.join(project_root, args.config)
    else:
        config_file_path = args.config

    if not os.path.exists(config_file_path):
        print(f"Error: Config file not found at {config_file_path}")
        sys.exit(1)

    preprocess_data(config_file_path) 