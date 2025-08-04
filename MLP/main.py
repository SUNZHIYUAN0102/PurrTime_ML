from matplotlib import ticker
import numpy as np
import pandas as pd
from sklearn.utils import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import seaborn as sns

from model import BalancedClassifier

# === 1. 加载并预处理数据 ===
df = pd.read_csv("collar_data5.csv")

feature_cols = [
    'X_Mean','X_Min','X_Max','X_Sum','X_sd','X_Skew','X_Kurt',
    'Y_Mean','Y_Min','Y_Max','Y_Sum','Y_sd','Y_Skew','Y_Kurt',
    'Z_Mean','Z_Min','Z_Max','Z_Sum','Z_sd','Z_Skew','Z_Kurt',
    'VM_Mean','VM_Min','VM_Max','VM_Sum','VM_sd','VM_Skew','VM_Kurt',
    'Cor_XY','Cor_XZ','Cor_YZ'
]

# 检查类别分布
print("原始数据类别分布:")
behavior_counts = df["Behaviour"].value_counts()
print(behavior_counts)
print(f"总样本数: {len(df)}")
print(f"类别数: {len(behavior_counts)}")
print()

X = df[feature_cols].values
y = df["Behaviour"].values

# 标签编码
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 特征标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# === 2. 分层拆分数据集 ===
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
)

print(f"训练集大小: {len(X_train)}")
print(f"验证集大小: {len(X_val)}")
print(f"测试集大小: {len(X_test)}")

# === 3. 创建 Dataset 和 DataLoader ===
class BehaviorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = BehaviorDataset(X_train, y_train)
val_dataset = BehaviorDataset(X_val, y_val)
test_dataset = BehaviorDataset(X_test, y_test)

# 使用更大的batch size减少噪声
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False)

# === 4. 平衡的神经网络模型 ===

input_dim = len(feature_cols)
num_classes = len(label_encoder.classes_)
model = BalancedClassifier(input_dim, num_classes)

print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# === 5. 损失函数和优化器 ===
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# 平衡的学习率和正则化
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 温和的学习率调度
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, patience=8
)

# === 6. 训练模型 - 严格的早停策略 ===
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

best_val_acc = 0.0
early_stop_patience = 15  # 适中的耐心值
patience_counter = 0
min_epochs = 15  # 减少最少训练轮数

print("开始训练...")
for epoch in range(100):  # 减少最大epochs
    # --- Train ---
    model.train()
    total_train_loss = 0
    correct_train = 0
    total_train = 0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        # 温和的梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == y_batch).sum().item()
        total_train += y_batch.size(0)

    train_accuracy = correct_train / total_train

    # --- Validation ---
    model.eval()
    total_val_loss = 0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            if X_batch.size(0) == 1:
                continue
                
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == y_batch).sum().item()
            total_val += y_batch.size(0)

    val_accuracy = correct_val / total_val
    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0

    # 学习率调度
    scheduler.step(avg_val_loss)

    # Early stopping - 基于验证准确率，但也考虑过拟合程度
    overfitting_gap = train_accuracy - val_accuracy
    
    if val_accuracy > best_val_acc and overfitting_gap < 0.20:  # 放宽过拟合限制
        best_val_acc = val_accuracy
        patience_counter = 0
        torch.save(model.state_dict(), "best_simple_model.pt")
    else:
        patience_counter += 1

    # --- Logging ---
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    if (epoch + 1) % 5 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:03}, Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Train Acc: {train_accuracy:.2%}, "
              f"Val Acc: {val_accuracy:.2%}, Gap: {overfitting_gap:.2%}, LR: {current_lr:.6f}")

    # 检查过拟合并提前警告
    if epoch > min_epochs:
        if overfitting_gap > 0.2:  # 过拟合严重
            print(f"警告: 严重过拟合 (Gap: {overfitting_gap:.2%})")
        
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1}, best val acc: {best_val_acc:.2%}")
            break

# 加载最佳模型进行测试
model.load_state_dict(torch.load("best_simple_model.pt"))

# === 7. 测试模型 ===
print(f"\n最佳验证准确率: {best_val_acc:.2%}")

model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        if X_batch.size(0) == 1:
            continue
            
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

test_accuracy = correct / total
print(f"🎯 最终测试准确率: {test_accuracy:.2%}")

from sklearn.metrics import classification_report, f1_score

print("\n📊 详细分类报告：")
report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_, output_dict=True)
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

f1_macro = f1_score(all_labels, all_preds, average='macro')
f1_weighted = f1_score(all_labels, all_preds, average='weighted')
print(f"\n🎯 Macro F1-score: {f1_macro:.4f}")
print(f"🎯 Weighted F1-score: {f1_weighted:.4f}")

# === 混淆矩阵 ===
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# === 8. 绘制训练曲线 ===
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_losses, "o", label="Train Loss", alpha=0.7)
plt.plot(val_losses, label="Val Loss", alpha=0.7)
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim(0.6, 1.1)
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(train_accuracies, "o", label="Train Accuracy", alpha=0.7)
plt.plot(val_accuracies, label="Val Accuracy", alpha=0.7)
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0.3, 0.8)
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))
plt.legend()
plt.grid(True)

# 过拟合程度图
plt.subplot(1, 3, 3)
gap = np.array(train_accuracies) - np.array(val_accuracies)
plt.plot(gap, label="Overfitting Gap", color='red', alpha=0.7)
plt.axhline(y=0.1, color='orange', linestyle='--', label='Acceptable Gap (10%)')
plt.axhline(y=0.2, color='red', linestyle='--', label='Severe Overfitting (20%)')
plt.title("Overfitting Analysis")
plt.xlabel("Epoch")
plt.ylabel("Train Acc - Val Acc")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")