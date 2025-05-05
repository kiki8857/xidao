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
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

# Import necessary functions from src
try:
    from src.models.random_forest import build_random_forest
    from src.training.evaluate import calculate_regression_metrics, check_metrics_thresholds
    from src.utils.logger import setup_logger
except ImportError as e:
    raise ImportError(f"Could not import project modules. Ensure PYTHONPATH is set correctly or run from root: {e}")

def train_evaluate_rf(config_path):
    """Loads features, trains/evaluates RF model, and saves results."""

    # 1. Load configuration
    print(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        return

    # --- Create results directory for this run ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.abspath(os.path.join(project_root, 'results', f'random_forest_{timestamp}'))
    os.makedirs(results_dir, exist_ok=True)
    log_file_path = os.path.join(results_dir, 'rf_training.log') # Log inside results dir

    # --- Configure Logger ---
    log_level_str = config.get('logging', {}).get('log_level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger = setup_logger('RFTrainingLogger', log_level, log_file_path)
    logger.info(f"Logger setup complete. Log level: {log_level_str}. Log file: {log_file_path}")
    logger.info(f"Results will be saved in: {results_dir}")
    logger.info(f"Using configuration from: {config_path}")

    # --- Set Random Seed ---
    random_seed = config.get('training', {}).get('random_state', None)
    if random_seed is not None:
        logger.info(f"Setting random seed to: {random_seed}")
        np.random.seed(random_seed)

    # --- Get Processed Data Paths ---
    processed_data_dir_rel = config.get('data', {}).get('processed_data_dir', 'data/processed')
    processed_data_dir = os.path.abspath(os.path.join(project_root, processed_data_dir_rel))
    features_path = os.path.join(processed_data_dir, 'final_features.csv')
    target_path = os.path.join(processed_data_dir, 'target_wear.csv')

    # 2. Load Final Features and Target
    logger.info("--- Step 1: Loading Final Features and Target ---")
    try:
        # Load features, ensuring the multi-index is parsed correctly
        final_features_df = pd.read_csv(features_path, index_col=['experiment_id', 'measurement_id'])
        # Load target, ensuring index is parsed and it becomes a Series
        target_series = pd.read_csv(target_path, index_col=['experiment_id', 'measurement_id']).squeeze("columns")
        if not isinstance(target_series, pd.Series):
             raise ValueError("Loaded target is not a Pandas Series after squeeze.")
        logger.info(f"Loaded features shape: {final_features_df.shape}")
        logger.info(f"Loaded target shape: {target_series.shape}")
    except FileNotFoundError:
        logger.error(f"Features file ({features_path}) or target file ({target_path}) not found. Run engineer_features.py first.")
        return
    except Exception as e:
        logger.error(f"Error loading features or target: {e}", exc_info=True)
        return
        
    # Check alignment
    if not final_features_df.index.equals(target_series.index):
         logger.warning("Index mismatch between loaded features and target. Attempting to align.")
         common_index = final_features_df.index.intersection(target_series.index)
         if common_index.empty:
              logger.error("No common index found between features and target after loading. Aborting.")
              return
         final_features_df = final_features_df.loc[common_index]
         target_series = target_series.loc[common_index]
         logger.info(f"Aligned data shapes: Features={final_features_df.shape}, Target={target_series.shape}")


    # 3. Split Data (Train/Test based on experiment ID)
    logger.info("--- Step 2: Splitting Data (Train/Test based on experiment ID) ---")
    train_exp_ids = config.get('training', {}).get('train_experiments', ['c1', 'c4'])
    test_exp_ids = config.get('training', {}).get('test_experiments', ['c6'])

    available_exp = final_features_df.index.get_level_values('experiment_id').unique().tolist()
    train_exp_ids = [eid for eid in train_exp_ids if eid in available_exp]
    test_exp_ids = [eid for eid in test_exp_ids if eid in available_exp]

    if not train_exp_ids:
        logger.error(f"No training experiments found in the data ({available_exp}). Configured: {config.get('training', {}).get('train_experiments')}. Aborting.")
        return
    if not test_exp_ids:
        logger.warning(f"No testing experiments found ({available_exp}). Configured: {config.get('training', {}).get('test_experiments')}. Evaluation will be skipped.")

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

    # 4. Build Random Forest Model
    logger.info("--- Step 3: Building Random Forest Model ---")
    model = build_random_forest(config, random_state=random_seed)
    if model is None:
        logger.error("Failed to build Random Forest model. Aborting.")
        return

    # 5. Train Model
    logger.info("--- Step 4: Training Random Forest Model ---")
    try:
        model.fit(X_train, y_train)
        logger.info("Random Forest model training complete.")
        # Save RF model
        model_save_path = os.path.join(results_dir, 'random_forest_model.joblib')
        joblib.dump(model, model_save_path)
        logger.info(f"Trained Random Forest model saved to {model_save_path}")
    except Exception as e:
        logger.error(f"Error training Random Forest: {e}", exc_info=True)
        return

    # 6. Evaluate Model on Test Set
    logger.info("--- Step 5: Evaluating Model on Test Set ---")
    if not X_test.empty and not y_test.empty:
        try:
            y_pred = model.predict(X_test)
            metrics = calculate_regression_metrics(y_test, y_pred)
            logger.info(f"Test Set Metrics: {metrics}")

            # Check thresholds
            thresholds = config.get('training', {}).get('target_metrics', {})
            passed_thresholds = check_metrics_thresholds(metrics, thresholds)
            logger.info(f"Threshold check result: {passed_thresholds}")
            metrics['passed_thresholds'] = passed_thresholds # Add result to saved metrics

            # Save evaluation results
            eval_save_path = os.path.join(results_dir, 'evaluation_results.yaml')
            with open(eval_save_path, 'w') as f:
                yaml.dump(metrics, f, default_flow_style=False)
            logger.info(f"Evaluation metrics saved to {eval_save_path}")

            # Save test predictions
            predictions_df = pd.DataFrame({'actual': y_test, 'predicted': y_pred}, index=X_test.index)
            pred_save_path = os.path.join(results_dir, 'test_predictions.csv')
            predictions_df.to_csv(pred_save_path, index=True)
            logger.info(f"Test predictions saved to {pred_save_path}")

        except Exception as e:
            logger.error(f"Error during model evaluation or saving results: {e}", exc_info=True)
    else:
        logger.warning("Test set is empty. Skipping evaluation.")

    # Save used config copy
    try:
         with open(os.path.join(results_dir, 'config_used.yaml'), 'w') as f:
             yaml.dump(config, f, default_flow_style=False)
    except Exception as dump_e:
         logger.warning(f"Could not save config copy: {dump_e}")
         
    logger.info(f"--- Random Forest Training and Evaluation Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate Random Forest Model")
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

    train_evaluate_rf(config_file_path) 