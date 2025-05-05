import logging
import sys
import os
from datetime import datetime

def setup_logger(name='ToolWearLogger', log_level=logging.INFO, log_file=None):
    """配置并返回一个日志记录器。

    Args:
        name (str): 日志记录器的名称。
        log_level (int): 日志记录级别 (例如 logging.INFO, logging.DEBUG)。
        log_file (str, optional): 日志文件的完整路径。如果提供，日志将同时输出到控制台和文件。
                                Defaults to None (只输出到控制台)。

    Returns:
        logging.Logger: 配置好的日志记录器实例。
    """
    logger = logging.getLogger(name)
    
    # 防止重复添加 handler (如果函数被多次调用)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(log_level)

    # 定义日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')

    # --- 控制台 Handler ---
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # --- 文件 Handler (如果指定了路径) ---
    if log_file:
        try:
            # 确保日志文件目录存在
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, mode='a') # 追加模式
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging to console and file: {log_file}")
        except Exception as e:
            logger.error(f"Failed to set up file logging to {log_file}: {e}", exc_info=True)
            # 即使文件设置失败，控制台日志仍然可用
    else:
        logger.info("Logging to console only.")

    return logger

# # --- Example Usage ---
# if __name__ == '__main__':
#     # 1. 基本用法 (仅控制台)
#     print("--- Basic Console Logging --- (Level: INFO)")
#     basic_logger = setup_logger('BasicLogger', log_level=logging.INFO)
#     basic_logger.debug("This debug message won't show.")
#     basic_logger.info("This is an info message.")
#     basic_logger.warning("This is a warning.")
#     basic_logger.error("This is an error.")

#     # 2. 输出到文件 (控制台 + 文件)
#     print("\n--- File and Console Logging --- (Level: DEBUG)")
#     log_filename = f'test_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
#     # 将日志文件放在脚本所在目录的 logs 子目录中
#     log_path = os.path.join(os.path.dirname(__file__), 'logs', log_filename)
    
#     file_logger = setup_logger('FileLogger', log_level=logging.DEBUG, log_file=log_path)
#     file_logger.debug(f"This debug message WILL show. Log file: {log_path}")
#     file_logger.info("Another info message.")
#     file_logger.error("Another error message.")
    
#     print(f"\nCheck the log file created at: {log_path}")

#     # 3. 再次获取 logger (验证是否会重复添加 handler)
#     print("\n--- Getting Logger Again ---")
#     same_file_logger = setup_logger('FileLogger') # 使用相同的名字
#     same_file_logger.info("This message should appear only once per handler in the log.")


