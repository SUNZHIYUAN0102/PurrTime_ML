import torch
import torch.nn as nn
import numpy as np
import joblib

from model import BalancedClassifier

# === 加载 scaler 和 label_encoder ===
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# === 定义输入维度和类别数量 ===
input_dim = scaler.mean_.shape[0]
num_classes = len(label_encoder.classes_)

# === 初始化并加载模型 ===
model = BalancedClassifier(input_dim, num_classes)
model.load_state_dict(torch.load("best_simple_model.pt"))
model.eval()

print("✅ 模型与预处理器加载成功！")

# === 预测函数 ===
def predict(sample_input):
    """
    参数: sample_input: List 或 1D np.array，长度等于特征数量
    返回: 预测的行为名称
    """
    if isinstance(sample_input, list):
        sample_input = np.array(sample_input)

    input_scaled = scaler.transform([sample_input])
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    predicted_label = label_encoder.inverse_transform(predicted.cpu().numpy())[0]
    return predicted_label

sample_input = [0.0862333333333333,-0.176,0.215,2.587,0.104884068456997,-1.00203964886314,3.24429471201925,-0.8002,-1.039,-0.605,-24.006,0.0904709517107468,-0.983923462454335,4.39036454382689,0.578266666666667,0.453,0.691,17.348,0.0616759794148549,-0.239311053852577,2.29609339915702,0.997857380489313,0.857410636742979,1.25104036705456,29.9357214146794,0.0946088947648492,1.02462091900498,3.6988884678451,-0.141102610904631,0.166640722824915,-0.486275046333345] 
predicted_behavior = predict(sample_input)
print(f"预测的行为: {predicted_behavior}")