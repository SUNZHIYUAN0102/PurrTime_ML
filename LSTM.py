import torch.nn as nn
import torch.nn.functional as F

class ImprovedLSTMClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=64, num_layers=2, dropout=0.4):
        super().__init__()
        # 减少模型复杂度
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,  # 从128减少到64
            num_layers=num_layers,  
            batch_first=True,
            dropout=0.0,  # LSTM内部dropout设为0，用其他方式控制
            bidirectional=False
        )
        
        # 添加更多正则化层
        self.dropout1 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # 添加中间层减少过拟合
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        # 使用更保守的权重初始化
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                elif 'bn' in name:  # BatchNorm权重
                    nn.init.ones_(param)
                elif len(param.shape) >= 2:  # 只对2维以上的权重使用xavier
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        # x: (B, S, F)
        out, _ = self.lstm(x)        # out: (B, S, H)
        out = out[:, -1, :]          # 取最后时刻: (B, H)
        
        # 添加正则化
        out = self.dropout1(out)
        out = self.bn1(out)
        out = F.relu(out)
        
        # 中间层
        out = self.fc1(out)
        out = self.dropout2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        # 输出层
        out = self.fc2(out)
        return out