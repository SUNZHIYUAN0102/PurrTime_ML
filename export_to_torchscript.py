# export_to_torchscript.py

import torch
import joblib
from model import BalancedClassifier  # 确保路径正确

# === 加载 scaler 和 label_encoder（获取维度信息）===
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

input_dim = scaler.mean_.shape[0]
num_classes = len(label_encoder.classes_)

# === 初始化模型并加载训练好的权重 ===
model = BalancedClassifier(input_dim, num_classes)
model.load_state_dict(torch.load("best_simple_model.pt"))
model.eval()  # 非常重要，会禁用 Dropout 和设置 BatchNorm 为 eval 模式

# === 创建一个假输入，trace 模型结构 ===
example_input = torch.randn(1, input_dim)
traced_model = torch.jit.trace(model, example_input)

# === 保存为 TorchScript 格式 ===
traced_model.save("model.pt")

print("✅ TorchScript 模型保存成功为 model.pt")
