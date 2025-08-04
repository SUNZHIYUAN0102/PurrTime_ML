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

sample_input = [0.2418,0.195,0.293,7.254,0.0299705602676797,-0.151553195839145,1.76843790565362,0.645133333333333,0.555,0.699,19.354,0.0409058914341564,-0.659851882859082,2.34928375698189,-0.721966666666667,-0.793,-0.684,-21.659,0.0282262818753329,-0.783847488031167,2.88569175424586,0.999425272267263,0.951203973919369,1.05316048159813,29.9827581680179,0.0177693601123114,0.406763052435147,5.39041048394318,-0.77678516720053,-0.494351086029931,0.642869930186697] 
predicted_behavior = predict(sample_input)
print(f"预测的行为: {predicted_behavior}")