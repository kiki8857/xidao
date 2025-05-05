#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from datetime import datetime

# 添加项目根目录到系统路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入调优模块
from src.training.tuning import tune_hyperparameters
from src.utils.logger import setup_logger

def main():
    """执行超参数调优脚本"""
    parser = argparse.ArgumentParser(description="调用超参数调优流程进行模型优化")
    parser.add_argument('--config', type=str, default='config/config.yaml',
                      help="配置文件路径（默认：config/config.yaml）")
    parser.add_argument('--model', type=str, choices=['random_forest', 'bpnn'], 
                      help="指定要调优的模型类型：random_forest 或 bpnn")
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"错误：配置文件未找到：{args.config}")
        sys.exit(1)
    
    # 设置日志记录器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(project_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'tuning_{args.model}_{timestamp}.log')
    logger = setup_logger('TuningScript', 'INFO', log_file)
    
    # 如果指定了模型类型，暂时修改配置中的模型类型
    if args.model:
        import yaml
        logger.info(f"指定调优模型类型为: {args.model}")
        # 读取配置文件
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # 备份原配置
        original_model_type = config.get('training', {}).get('model_type', 'random_forest')
        
        # 修改模型类型
        if 'training' not in config:
            config['training'] = {}
        config['training']['model_type'] = args.model
        
        # 创建临时配置文件
        temp_config_path = os.path.join(project_root, 'config', f'temp_config_{timestamp}.yaml')
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        logger.info(f"已创建临时配置文件: {temp_config_path}")
        config_path = temp_config_path
    else:
        config_path = args.config
    
    # 执行调优过程
    logger.info(f"开始执行超参数调优，使用配置文件: {config_path}")
    try:
        tune_hyperparameters(config_path)
        logger.info("超参数调优完成")
    except Exception as e:
        logger.error(f"调优过程中出现错误: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # 清理临时配置文件
        if args.model and os.path.exists(temp_config_path):
            try:
                os.remove(temp_config_path)
                logger.info(f"已删除临时配置文件: {temp_config_path}")
            except:
                logger.warning(f"无法删除临时配置文件: {temp_config_path}")

if __name__ == '__main__':
    main() 