from typing import Counter
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

from LSTM import ImprovedLSTMClassifier

# === 1. 加载并预处理数据 ===
df = pd.read_csv("collar_data5.csv")
df = df.sort_values(["Cat_id", "Timestamp"]).reset_index(drop=True)

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
df["y_enc"] = label_encoder.fit_transform(df["Behaviour"])

cat_col = 'Cat_id'
df_no_jellyb = df[df[cat_col] != 'JellyB']

# 所有猫 ID 列表
all_cats = df_no_jellyb[cat_col].unique()

# 随机打乱（固定种子保证可复现）
rng = np.random.default_rng(42)
rng.shuffle(all_cats)

# 划分：7 只训练，2 只验证，剩下 2 只测试
train_cats = all_cats[:7]
val_cats   = all_cats[7:9]
test_cats  = all_cats[9:11]

print("Train cats:", train_cats)
print("Val cats:", val_cats)
print("Test cats:", test_cats)

# 建立索引掩码（基于原 df 行顺序）
mask_train = (df[cat_col].isin(train_cats)) & (df[cat_col] != 'JellyB')
mask_val   = (df[cat_col].isin(val_cats))
mask_test  = (df[cat_col].isin(test_cats))

scaler = StandardScaler()
scaler.fit(df.loc[mask_train, feature_cols].values)

X_all = scaler.transform(df[feature_cols].values)
y_all = df["y_enc"].values
cats_all = df[cat_col].values


def make_seq_windows(X, y, cats, window_size=10, stride=1, label_mode="majority", min_purity=0.7):
    Xs, ys = [], []
    for cat in np.unique(cats):
        idx = np.where(cats == cat)[0]
        Xi, yi = X[idx], y[idx]
        n = len(idx)
        for s in range(0, n - window_size + 1, stride):
            segy = yi[s:s+window_size]
            vals, cnts = np.unique(segy, return_counts=True)
            maj = vals[np.argmax(cnts)]
            purity = cnts.max() / window_size
            if purity < min_purity:   # 丢掉过渡窗口
                continue
            Xs.append(Xi[s:s+window_size])
            ys.append(maj)
    return np.array(Xs, np.float32), np.array(ys, np.int64)


X_train_seq, y_train = make_seq_windows(
    X_all[mask_train], y_all[mask_train], cats_all[mask_train],
    window_size=12, stride=4
)
X_val_seq,   y_val   = make_seq_windows(
    X_all[mask_val], y_all[mask_val], cats_all[mask_val],
    window_size=12, stride=4
)
X_test_seq,  y_test  = make_seq_windows(
    X_all[mask_test], y_all[mask_test], cats_all[mask_test],
    window_size=12, stride=4
)

counter = Counter(y_train)
total = len(y_train)

print("📊 窗口级训练集类别分布：")
for label_idx in sorted(counter.keys()):
    count = counter[label_idx]
    ratio = count / total * 100
    print(f"{label_encoder.classes_[label_idx]}: {count} ({ratio:.2f}%)")


def balance_windows_offline(X_seq, y, mode="cap_majority", cap_per_class=3000, target_per_class=None, seed=42):
    """
    X_seq: (N, S, F)
    y:     (N,)
    mode:
      - "cap_majority": 对每一类最多保留 cap_per_class（欠采样多数类）
      - "target_equal": 把每一类都采样到 target_per_class（少的过采样，多的欠采样）
    """
    rng = np.random.default_rng(seed)

    X_out, y_out = [], []
    classes = np.unique(y)

    if mode == "cap_majority":
        for c in classes:
            idx = np.where(y == c)[0]
            if len(idx) > cap_per_class:
                idx = rng.choice(idx, size=cap_per_class, replace=False)
            X_out.append(X_seq[idx])
            y_out.append(y[idx])

    elif mode == "target_equal":
        if target_per_class is None:
            raise ValueError("target_per_class must be set when mode='target_equal'")
        for c in classes:
            idx = np.where(y == c)[0]
            if len(idx) >= target_per_class:
                idx = rng.choice(idx, size=target_per_class, replace=False)
            else:
                extra = rng.choice(idx, size=target_per_class - len(idx), replace=True)  # 过采样
                idx = np.concatenate([idx, extra], axis=0)
            X_out.append(X_seq[idx])
            y_out.append(y[idx])
    else:
        raise ValueError("Unknown mode")

    Xb = np.concatenate(X_out, axis=0)
    yb = np.concatenate(y_out, axis=0)

    # 打乱
    perm = rng.permutation(len(yb))
    return Xb[perm], yb[perm]

# 用法1：对多数类做“最多8000窗口”的欠采样
X_train_seq, y_train = balance_windows_offline(
    X_train_seq, y_train,
    # mode="target_equal", target_per_class=5000
)
counter = Counter(y_train)
total = len(y_train)

print("📊 采样训练集类别分布：")
for label_idx in sorted(counter.keys()):
    count = counter[label_idx]
    ratio = count / total * 100
    print(f"{label_encoder.classes_[label_idx]}: {count} ({ratio:.2f}%)")

# === 3. 创建 Dataset 和 DataLoader ===
class BehaviorSeqDataset(Dataset):
    def __init__(self, X_seq, y):
        self.X = torch.from_numpy(X_seq)          # (N, S, F), float32
        self.y = torch.from_numpy(y).long()       # (N,)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = BehaviorSeqDataset(X_train_seq, y_train)
val_dataset   = BehaviorSeqDataset(X_val_seq,   y_val)
test_dataset  = BehaviorSeqDataset(X_test_seq,  y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, drop_last=False)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, drop_last=False)
# === 4. 平衡的神经网络模型 ===

input_dim = len(feature_cols)
num_classes = len(label_encoder.classes_)
model = ImprovedLSTMClassifier(input_dim=input_dim, num_classes=num_classes)
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

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

beta = 0.999
counts = np.bincount(y_train)
eff_num = (1 - np.power(beta, counts)) / (1 - beta)
alpha = eff_num.sum() / (eff_num + 1e-8)
alpha = alpha / alpha.sum()  # 归一化到和为1
alpha = torch.tensor(alpha, dtype=torch.float32)

focal_criterion = FocalLoss(alpha=alpha, gamma=2, reduction='mean')

criterion = focal_criterion

# 平衡的学习率和正则化
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4, amsgrad=True)

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
