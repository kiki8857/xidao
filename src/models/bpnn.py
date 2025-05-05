import torch
import torch.nn as nn
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)

class BPNN(nn.Module):
    """定义一个简单的 BP (多层感知器) 神经网络用于回归。"""
    def __init__(self, input_dim, hidden_layers, output_dim=1, activation='relu', output_activation='linear', dropout_rate=0, batch_norm=False):
        """
        Args:
            input_dim (int): 输入特征的数量。
            hidden_layers (list): 一个包含每个隐藏层神经元数量的列表。
            output_dim (int): 输出维度 (对于回归通常是 1)。
            activation (str): 隐藏层使用的激活函数 ('relu', 'leaky_relu', 'elu', 或 'tanh')。
            output_activation (str): 输出层的激活函数 ('linear', 'tanh', 'sigmoid'等)。
            dropout_rate (float): dropout比例，用于防止过拟合，范围0-1。
            batch_norm (bool): 是否使用批归一化。
        """
        super(BPNN, self).__init__()
        layers = OrderedDict()
        last_dim = input_dim

        # 添加隐藏层
        for i, hidden_dim in enumerate(hidden_layers):
            layers[f'linear_{i}'] = nn.Linear(last_dim, hidden_dim)
            
            # 添加批归一化（如果启用）
            if batch_norm:
                layers[f'batchnorm_{i}'] = nn.BatchNorm1d(hidden_dim)
            
            # 添加激活函数
            if activation.lower() == 'relu':
                layers[f'activation_{i}'] = nn.ReLU()
            elif activation.lower() == 'tanh':
                layers[f'activation_{i}'] = nn.Tanh()
            elif activation.lower() == 'leaky_relu':
                layers[f'activation_{i}'] = nn.LeakyReLU(0.1)
            elif activation.lower() == 'elu':
                layers[f'activation_{i}'] = nn.ELU()
            else:
                logger.warning(f"Unsupported activation '{activation}', defaulting to ReLU.")
                layers[f'activation_{i}'] = nn.ReLU()
                
            # 添加 Dropout（如果指定）
            if dropout_rate > 0:
                layers[f'dropout_{i}'] = nn.Dropout(p=dropout_rate)
                
            last_dim = hidden_dim

        # 添加输出层
        layers['output_linear'] = nn.Linear(last_dim, output_dim)
        
        # 添加输出层激活函数（如果需要）
        if output_activation.lower() != 'linear':
            if output_activation.lower() == 'tanh':
                layers['output_activation'] = nn.Tanh()
            elif output_activation.lower() == 'sigmoid':
                layers['output_activation'] = nn.Sigmoid()
            elif output_activation.lower() == 'relu':
                layers['output_activation'] = nn.ReLU()
            else:
                logger.warning(f"Unsupported output activation '{output_activation}', using linear (no activation).")

        self.network = nn.Sequential(layers)

    def forward(self, x):
        """定义前向传播路径。"""
        return self.network(x)

def build_bpnn(input_dim, config):
    """根据配置构建 BPNN 模型。

    Args:
        input_dim (int): 输入特征的维度。
        config (dict): 包含 models.bpnn 参数的配置字典。
                       例如 hidden_layers, activation, dropout_rate。

    Returns:
        BPNN: 一个 PyTorch BPNN 模型实例。
        None: 如果配置错误或输入维度无效。
    """
    if input_dim <= 0:
        logger.error(f"Invalid input_dim for BPNN: {input_dim}. Must be > 0.")
        return None
        
    bpnn_config = config.get('models', {}).get('bpnn', {})
    hidden_layers = bpnn_config.get('hidden_layers', [100, 50]) # 默认结构
    activation = bpnn_config.get('activation', 'relu').lower()
    output_activation = bpnn_config.get('output_activation', 'linear').lower()
    output_dim = bpnn_config.get('output_dim', 1) # 默认为1，用于回归
    dropout_rate = bpnn_config.get('dropout_rate', 0) # 默认不使用dropout
    batch_norm = bpnn_config.get('batch_normalization', False) # 默认不使用批归一化
    
    logger.info(f"Building BPNN with: input_dim={input_dim}, hidden_layers={hidden_layers}, activation='{activation}', output_activation='{output_activation}', dropout={dropout_rate}, batch_norm={batch_norm}, output_dim={output_dim}")

    try:
        model = BPNN(input_dim, hidden_layers, output_dim, activation, output_activation, dropout_rate, batch_norm)
        return model
    except Exception as e:
        logger.error(f"Failed to build BPNN model: {e}", exc_info=True)
        return None

# # --- Example Usage ---
# if __name__ == '__main__':
#     import yaml
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#     # 1. Load Config
#     config_path = '../../config/config.yaml' # Adjust path
#     try:
#         with open(config_path, 'r') as f:
#             config = yaml.safe_load(f)
#     except Exception as e:
#         print(f"Error loading config: {e}")
#         config = {}

#     # 2. Define Input Dimension (e.g., after PCA)
#     # 这个值需要根据数据处理步骤的输出确定
#     example_input_dim = 10 # 假设 PCA 后得到 10 个特征

#     # 3. Build Model
#     print("Building BPNN model...")
#     bpnn_model = build_bpnn(example_input_dim, config)

#     if bpnn_model:
#         print("\nBPNN model built successfully!")
#         print(bpnn_model)

#         # 打印模型参数数量 (示例)
#         total_params = sum(p.numel() for p in bpnn_model.parameters() if p.requires_grad)
#         print(f"\nTotal trainable parameters: {total_params:,}")

#         # 测试模型前向传播 (需要 dummy input)
#         try:
#             dummy_input = torch.randn(5, example_input_dim) # Batch of 5 samples
#             output = bpnn_model(dummy_input)
#             print(f"\nDummy forward pass output shape: {output.shape}") # Expected: [5, 1]
#             print("Dummy output:")
#             print(output)
#         except Exception as forward_e:
#              print(f"\nError during dummy forward pass: {forward_e}")

#     else:
#         print("\nFailed to build BPNN model.")
