'''
不生成任何
只显示图片
'''
import os
import time
import math
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# plt.rcParams['figure.figsize'] = (6.6, 6)  # 显示图像的最大范围
plt.rcParams['figure.figsize'] = (4.4, 3.3)

# 1. 定义神经网络结构
class SimpleNN(nn.Module):
    def __init__(self, input_size=5, hidden_size=3, hidden_size2=2, output_size=1):
        super(SimpleNN, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size)  # 隐藏层
        # self.hidden2 = nn.Linear(hidden_size, hidden_size2)  # 隐藏层
        self.output = nn.Linear(hidden_size2, output_size)  # 输出层

    def forward(self, x):
        x = torch.sigmoid(self.hidden1(x))  # 隐藏层激活函数使用Sigmoid
        # x = torch.sigmoid(self.hidden2(x))  # 隐藏层激活函数使用Sigmoid
        x = self.output(x)  # 输出层
        return x


'''预测'''
#%% function
def getName(signalPath):
    return signalPath[-2:]

def get_original_from_reshape(datax1):
    '''
    重塑还原
    :param datax1: ?x50x2,第一列loop，第二列HI值
    :return: ？x2
    '''
    # ================================ 优化显示: 拼接一个完整的HI趋势datay1_forPlot，第一列loop第二列HI值
    datay1_forPlot = np.zeros((datax1.shape[0]+50, 2))  # 构造数组
    for i in range(datax1.shape[0]):   # 数组的前半部分
        datay1_forPlot[i, 0] = datax1[i, 0, 0].squeeze(-1)
        datay1_forPlot[i, 1] = datax1[i, 0, 1].squeeze(-1)
    for i in range(datax1.shape[0], datax1.shape[0]+50):  # 数组的后半部分
        datay1_forPlot[i, 0] = datax1[-1, i-datax1.shape[0], 0].squeeze(-1)
        datay1_forPlot[i, 1] = datax1[-1, i-datax1.shape[0], 1].squeeze(-1)
    return datay1_forPlot
def computeMonotonicity(data):
    dH = np.diff(data)
    zheng = 0
    fu = 0
    for i in dH:
        if i >0:
            zheng +=1
        elif i<0:
            fu +=1
    s = (zheng-fu)/dH.shape[0]
    return s
def computer(a,b):
    if a.shape[0]>b.shape[0]:
        a = a[:b.shape[0]]
    else:
        b = b[:a.shape[0]]
    # 定义真实值数组和预测值数组
    # a = HI_after[:, 1]
    # b = valid_predict_data[:HI_after.shape[0]]
    # 均方误差（Mean Squared Error，MSE）
    mse = np.mean((a - b) ** 2)
    # 均方根误差（Root Mean Squared Error，RMSE）
    rmse = np.sqrt(mse)
    # 平均绝对误差（Mean Absolute Error，MAE）
    mae = np.mean(np.abs(a - b))
    # R平方（R-Squared，R^2）
    mean_a = np.mean(a)
    r_squared = 1 - np.sum((a - b) ** 2) / np.sum((a - mean_a) ** 2)
    # 平均百分比误差（Mean Percentage Error，MPE）
    mpe = np.mean(((a - b) / a) * 100)
    # 输出结果
    # print("均方误差（MSE）:", np.around(mse, 3))
    print("均方根误差（RMSE）:", np.around(rmse, 3))
    # print("平均绝对误差（MAE）:", np.around(mae, 3))
    print("R平方（R^2）:", np.around(r_squared, 3))
    # print("平均百分比误差（MPE）:", np.around(mpe, 3))
#%% main 画图
def plotPredictHI(save_pic_file ,read_data_path ,valid_dataset_path ,model_file ,device):
    test_data = np.load(valid_dataset_path)
    # rul = []
    model = torch.load(model_file).eval()  # 载入模型pkl
    # for i in range(test_data.shape[0]):
    test_x = torch.from_numpy(test_data)
    # rul = model.predict(test_data)
    # print(rul)
    with torch.no_grad():
        predictions = model(test_x)
    predictions_np = predictions.numpy()
    predictions_np = predictions_np.flatten().tolist()
    rul = [int(x) for x in predictions_np]
    print(rul)

#%%

device='cuda'  # device='cpu'

'''
D2 最后编号 151 7.669
D3 最后编号 174 7.43
D4 最后编号 136 7.47
D5 最后编号 162 7.572
'''

save_path = r'D:\A小论文\code\model_compare_code\PHM2010_winner\model\D1234-D5'
valid_dataset_path = save_path+'\\dataset\D5_x.npy'
num_i = 51

save_result_path = save_path+'\\simulation'
if not os.path.exists(save_result_path):
    os.mkdir(save_result_path)
read_data_path = save_path + '\\dataset'
model_file = save_path + "\\LearnRate\\0.827410_LSTM2FC.pkl"
save_pic_file = save_result_path + '\\' + str(num_i) + '.png'
RUL = plotPredictHI(save_pic_file, read_data_path, valid_dataset_path, model_file, device)
print(RUL)


