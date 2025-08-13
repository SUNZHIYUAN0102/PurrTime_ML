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
df = pd.read_csv("collar_data6.csv")

target_classes = [
    {"Behaviour": "Inactive", "Count": 30000},
    # {"Behaviour": "Grooming", "Count": 9000},
    # {"Behaviour": "Active", "Count": 6000}
]

# 处理采样
dfs = []
for item in target_classes:
    label = item["Behaviour"]
    target_count = item["Count"]
    class_df = df[df["Behaviour"] == label]
    current_count = len(class_df)
    
    if current_count > target_count:
        # 欠采样
        class_df = class_df.sample(n=target_count, random_state=42)

    dfs.append(class_df)

# 保留未处理类别的数据（如果有）
processed_labels = {item["Behaviour"] for item in target_classes}
unprocessed_df = df[~df["Behaviour"].isin(processed_labels)]
dfs.append(unprocessed_df)

# 合并 & 打乱
df = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)

# 结果打印
print("✅ 数据预处理完成！")
print(df['Behaviour'].value_counts())

feature_cols = [
    'X_Mean','X_Min','X_Max','X_Sum','X_sd','X_Skew','X_Kurt',
    'Y_Mean','Y_Min','Y_Max','Y_Sum','Y_sd','Y_Skew','Y_Kurt',
    'Z_Mean','Z_Min','Z_Max','Z_Sum','Z_sd','Z_Skew','Z_Kurt',
    'VM_Mean','VM_Min','VM_Max','VM_Sum','VM_sd','VM_Skew','VM_Kurt',
    'Cor_XY','Cor_XZ','Cor_YZ'
]

X = df[feature_cols].values
y = df["Behaviour"].values

# 标签编码
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# === 2. 分层拆分数据集 ===

cat_col = 'Cat_id'
df_no_jellyb = df[df[cat_col] != 'JellyB']
# df_no_jellyb = df.copy()

# 所有猫 ID 列表
all_cats = df_no_jellyb[cat_col].unique()

# 随机打乱（固定种子保证可复现）
rng = np.random.default_rng(55)
rng.shuffle(all_cats)

# 划分：6 只训练，2 只验证，剩下 2 只测试
train_cats = all_cats[:6]
val_cats   = all_cats[6:8]
test_cats  = all_cats[8:10]

print("Train cats:", train_cats)
print("Val cats:", val_cats)
print("Test cats:", test_cats)

# 训练集
train_mask = df_no_jellyb[cat_col].isin(train_cats)
X_train = df_no_jellyb.loc[train_mask, feature_cols].values
y_train = label_encoder.transform(df_no_jellyb.loc[train_mask, "Behaviour"].values)

# 验证集
val_mask = df_no_jellyb[cat_col].isin(val_cats)
X_val = df_no_jellyb.loc[val_mask, feature_cols].values
y_val = label_encoder.transform(df_no_jellyb.loc[val_mask, "Behaviour"].values)

# 测试集
test_mask = df_no_jellyb[cat_col].isin(test_cats)
X_test = df_no_jellyb.loc[test_mask, feature_cols].values
y_test = label_encoder.transform(df_no_jellyb.loc[test_mask, "Behaviour"].values)

target_counts = {
    # 0: 7500,  # Active
    # 1: 8000,  # Eating
    # 2: 10000,  # Grooming
    # 3: 15000   # Inactive
}

# 收集采样后的数据
X_resampled = []
y_resampled = []

for label in np.unique(y_train):
    idx = np.where(y_train == label)[0]
    current_count = len(idx)

    if label in target_counts:
        target_count = target_counts[label]
        if current_count > target_count:
            # 欠采样
            sampled_idx = np.random.choice(idx, size=target_count, replace=False)
        else:
            # 过采样（有放回）
            sampled_idx = np.random.choice(idx, size=target_count, replace=True)
    else:
        # 不进行采样，保留原始样本
        sampled_idx = idx
    
    X_resampled.append(X_train[sampled_idx])
    y_resampled.append(y_train[sampled_idx])

# 合并并打乱
X_train = np.vstack(X_resampled)
y_train = np.hstack(y_resampled)

shuffle_idx = np.random.permutation(len(X_train))
X_train = X_train[shuffle_idx]
y_train = y_train[shuffle_idx]
# 打印采样后的数据分布
print("✅ 采样后的训练数据分布：")
for i, count in enumerate(np.bincount(y_train)):
    print(f"{label_encoder.classes_[i]}: {count}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

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
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)
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

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        alpha: Tensor of shape (num_classes,) with weights per class
        gamma: focusing parameter
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        targets = targets.view(-1, 1)

        # Gather log-prob and prob of the correct class
        log_pt = log_probs.gather(1, targets).squeeze(1)
        pt = probs.gather(1, targets).squeeze(1)

        # If alpha is provided
        if self.alpha is not None:
            at = self.alpha.gather(0, targets.squeeze())
            log_pt = log_pt * at

        loss = -1 * (1 - pt) ** self.gamma * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

counts = np.bincount(y_train)
beta = 0.999  # 越接近1，越强调小类
effective_num = 1.0 - np.power(beta, counts)
weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)  # 大约 ∝ 1/(1-β^n)
weights = weights / weights.mean()  # 归一化到均值=1
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

focal_criterion = FocalLoss(alpha=class_weights_tensor, gamma=3, reduction='mean')

criterion = focal_criterion

# 平衡的学习率和正则化
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3, amsgrad=True)

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
early_stop_patience = 20  # 适中的耐心值
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
plt.ylim(0, 1)
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.2))
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(train_accuracies, "o", label="Train Accuracy", alpha=0.7)
plt.plot(val_accuracies, label="Val Accuracy", alpha=0.7)
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0.3, 1)
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