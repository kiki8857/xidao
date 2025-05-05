import pandas as pd
import os
import glob
import logging
import re # 正则表达式用于解析文件名

logger = logging.getLogger(__name__)

def load_phm2010_data(data_dir):
    """加载 PHM 2010 数据集。

    根据观察到的目录结构进行加载:
    - 遍历 data_dir 下的 c1, c4, c6 等目录。
    - 在每个 cX 目录下:
        - 读取 cX_wear.csv 获取磨损标签。
        - 进入 cX/cX/ 子目录，读取所有 c_X_YYY.csv 信号文件。
        - 将信号数据和磨损数据合并。

    Args:
        data_dir (str): 包含 PHM 2010 原始数据 c1, c4, c6 等子目录的根目录。
                        (例如: 'xidao/data/raw/PHM_2010/')

    Returns:
        dict: 一个字典，键是实验名称 (例如 'c1')，值是包含该实验
              合并后数据的 Pandas DataFrame (信号 + 磨损标签)。
        None: 如果找不到数据或加载失败。
    """
    datasets = {}
    logger.info(f"Attempting to load PHM 2010 data from root directory: {data_dir}")

    # 查找 c1, c4, c6 等实验目录 (忽略 c2, c3, c5 如果它们不存在或格式不同)
    # 调整 glob pattern 以匹配存在的实验目录
    experiment_dirs = [d for d in glob.glob(os.path.join(data_dir, 'c[146]')) if os.path.isdir(d)] # 仅加载 c1, c4, c6
    # experiment_dirs = [d for d in glob.glob(os.path.join(data_dir, 'c*')) if os.path.isdir(d)] # 加载所有 c* 目录

    if not experiment_dirs:
        logger.error(f"No experiment directories (c1, c4, c6, etc.) found in {data_dir}.")
        try:
            content = os.listdir(data_dir)
            logger.info(f"Contents of {data_dir}: {content}")
        except FileNotFoundError:
            logger.error(f"Directory not found: {data_dir}")
        return None

    # --- 定义信号列名 (根据数据集文档或数据简介确定) ---
    # 假设有7列: Fx, Fy, Fz, Vx, Vy, Vz, AE_RMS
    signal_column_names = ['Fx', 'Fy', 'Fz', 'Vx', 'Vy', 'Vz', 'AE_RMS']

    for exp_dir in experiment_dirs:
        experiment_name = os.path.basename(exp_dir)
        logger.info(f"Processing experiment: {experiment_name}")

        try:
            # 1. 加载磨损数据
            wear_file_path = os.path.join(exp_dir, f'{experiment_name}_wear.csv')
            if not os.path.exists(wear_file_path):
                logger.warning(f"Wear file not found for {experiment_name}: {wear_file_path}. Skipping experiment.")
                continue

            # --- 根据 c1_wear.csv 确认的结构直接读取 --- #
            try:
                wear_df = pd.read_csv(wear_file_path)
                # 确认包含 'cut' 和 flute 列
                required_cols = ['cut', 'flute_1', 'flute_2', 'flute_3']
                if not all(col in wear_df.columns for col in required_cols):
                    raise ValueError(f"Wear file {wear_file_path} missing required columns: {required_cols}. Found: {wear_df.columns.tolist()}")

                # 重命名 'cut' 列
                wear_df = wear_df.rename(columns={'cut': 'measurement_id'})

                # 计算平均磨损值
                flute_cols = ['flute_1', 'flute_2', 'flute_3']
                wear_df['wear_value'] = wear_df[flute_cols].mean(axis=1)

                # 只保留需要的列
                wear_df = wear_df[['measurement_id', 'wear_value']]
                logger.info(f"Loaded and processed wear data for {experiment_name}. Calculated average wear from flutes. Shape: {wear_df.shape}")

            except Exception as e:
                logger.error(f"Failed to read or process wear file {wear_file_path} with confirmed structure: {e}. Skipping experiment {experiment_name}.")
                continue

            # 检查 wear_df 是否成功加载和处理
            if wear_df is None or wear_df.empty:
                 logger.error(f"Wear dataframe is empty or None after attempting load for {experiment_name}. Skipping experiment.")
                 continue

            # -- 将 wear_df 的 measurement_id 设为索引，方便后续 map --
            try:
                wear_df = wear_df.astype({'measurement_id': int, 'wear_value': float})
                wear_map = wear_df.set_index('measurement_id')['wear_value']
            except Exception as map_e:
                logger.error(f"Error processing wear_df for mapping in {experiment_name}: {map_e}. Skipping experiment.")
                continue

            # 2. 加载信号数据
            signal_data_sub_dir = os.path.join(exp_dir, experiment_name)
            signal_files = sorted(glob.glob(os.path.join(signal_data_sub_dir, f'{experiment_name[0]}_{experiment_name[1]}_*.csv')))
            # 改进排序：按文件名中的数字排序
            try:
                 signal_files = sorted(signal_files, key=lambda x: int(re.search(r'_(\d+)\.csv$', x).group(1)))
            except AttributeError:
                 logger.warning(f"Could not parse numbers from filenames in {signal_data_sub_dir} for sorting. Using default sort.")
            except Exception as sort_e:
                 logger.warning(f"Error sorting signal files in {signal_data_sub_dir}: {sort_e}. Using default sort.")


            if not signal_files:
                logger.warning(f"No signal CSV files found in {signal_data_sub_dir}. Skipping experiment.")
                continue

            all_signals_list = []
            measurement_indices = []
            for i, signal_file in enumerate(signal_files):
                try:
                    # 解析文件名获取测量编号 (假设与 wear_df 的行号对应, 从 1 开始?)
                    match = re.search(r'_(\d+)\.csv$', signal_file)
                    measurement_id = int(match.group(1)) if match else i + 1 # 如果解析失败，使用索引+1

                    # 读取信号文件，假设没有表头
                    signal_df_single_measurement = pd.read_csv(signal_file, header=None, names=signal_column_names)

                    # 添加测量编号和实验ID列
                    signal_df_single_measurement['measurement_id'] = measurement_id
                    signal_df_single_measurement['experiment_id'] = experiment_name
                    all_signals_list.append(signal_df_single_measurement)
                    measurement_indices.append(measurement_id)

                except Exception as e:
                    logger.error(f"Failed to load or process signal file {signal_file}: {e}")
                    # 可以选择跳过此文件

            if not all_signals_list:
                logger.warning(f"No signal data successfully loaded for {experiment_name}. Skipping experiment.")
                continue

            # 合并所有信号测量到一个 DataFrame
            full_signal_df = pd.concat(all_signals_list, ignore_index=True)
            logger.info(f"Concatenated signal data for {experiment_name} with shape {full_signal_df.shape}")

            # 3. 合并信号与磨损数据
            # 现在使用 wear_map (以 measurement_id 为索引) 来添加 wear 列
            try:
                full_signal_df['wear'] = full_signal_df['measurement_id'].map(wear_map)
                final_df = full_signal_df
                # 检查是否有 NaN 值，这可能表示映射失败或 wear 文件不包含所有测量
                nan_wear_count = final_df['wear'].isnull().sum()
                if nan_wear_count > 0:
                     missing_ids = final_df[final_df['wear'].isnull()]['measurement_id'].unique()
                     logger.warning(f"{nan_wear_count} signal entries in {experiment_name} could not be mapped to a wear value. Missing IDs might include: {missing_ids[:5]}... (Total missing: {len(missing_ids)}) Wear map index min/max: {wear_map.index.min()}/{wear_map.index.max() if not wear_map.empty else 'N/A'}")
                     # 决定如何处理 NaN: 保留? 填充? 删除?
                     # 暂时保留 NaN
            except KeyError as map_key_e:
                 logger.error(f"KeyError during wear mapping for {experiment_name}: {map_key_e}. This likely means 'measurement_id' column is missing or incorrect in signal data. Skipping wear merge.")
                 final_df = full_signal_df # 返回没有 wear 列的数据
            except Exception as merge_e:
                 logger.error(f"Unexpected error during wear merging for {experiment_name}: {merge_e}. Skipping wear merge.")
                 final_df = full_signal_df

            datasets[experiment_name] = final_df
            logger.info(f"Successfully processed data for {experiment_name}. Final shape: {final_df.shape}")

        except Exception as exp_e:
            logger.error(f"Failed to process experiment {experiment_name}: {exp_e}", exc_info=True)
            #可以选择跳过此实验或中止

    if not datasets:
        logger.warning("No datasets were successfully loaded after processing all experiments.")
        return None

    logger.info(f"Successfully loaded and processed data for experiments: {list(datasets.keys())}")
    # 可以选择将所有实验合并到一个 DataFrame
    # combined_df = pd.concat(datasets.values(), ignore_index=True)
    # return combined_df
    return datasets

# # --- 示例用法 --- 
# if __name__ == '__main__':
#     import yaml
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#     # 加载配置
#     config_path = '../../config/config.yaml' # 相对于当前文件的路径
#     try:
#         with open(config_path, 'r') as f:
#             config = yaml.safe_load(f)
#         root_data_dir = config['data']['dataset_path']
#         print(f"Using data directory from config: {root_data_dir}")
#     except FileNotFoundError:
#         print(f"Config file not found at {config_path}. Attempting default relative path.")
#         root_data_dir = '../../data/raw/PHM_2010/' # 注意相对路径
#     except KeyError:
#          print("Config file missing 'data.dataset_path'. Using default relative path.")
#          root_data_dir = '../../data/raw/PHM_2010/'
#     except Exception as e:
#         print(f"Error loading config: {e}")
#         root_data_dir = '../../data/raw/PHM_2010/'

#     # 修正路径，确保它是相对于项目根目录的
#     # 如果脚本是从 src/data_processing 运行，向上两级到 xidao，然后进入 data/raw/PHM_2010
#     # 如果是从项目根目录 (bishe) 运行，路径应为 'xidao/data/raw/PHM_2010/'
#     # 假设我们总是从 src/ 目录内部运行脚本进行测试
#     script_dir = os.path.dirname(__file__)
#     project_root = os.path.abspath(os.path.join(script_dir, '../../')) # 找到 xidao 目录
#     if not root_data_dir.startswith('/'): # 如果不是绝对路径
#         root_data_dir_abs = os.path.abspath(os.path.join(project_root, root_data_dir)) 
#     else:
#         root_data_dir_abs = root_data_dir
#     print(f"Absolute data directory path: {root_data_dir_abs}")

#     loaded_data_dict = load_phm2010_data(root_data_dir_abs)

#     if loaded_data_dict:
#         print(f"\nSuccessfully loaded {len(loaded_data_dict)} experiments.")
#         for exp_name, df in loaded_data_dict.items():
#             print(f"\n--- Experiment: {exp_name} ---")
#             print(f"Shape: {df.shape}")
#             print("Columns:", df.columns.tolist())
#             print("First 5 rows:")
#             print(df.head())
#             print("Last 5 rows:")
#             print(df.tail())
#             # 检查 wear 列是否有 NaN
#             if 'wear' in df.columns:
#                 print(f"Wear column NaN count: {df['wear'].isnull().sum()}")
#             else:
#                 print("Wear column not found/added.")
#     else:
#         print("\nData loading failed.")
