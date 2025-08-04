import torch.nn as nn

class BalancedClassifier(nn.Module):
    def __init__(self, input_dim, num_classes): 
        super().__init__()
        self.net = nn.Sequential(
            # 第一层
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  # 适中的dropout

            # 第二层
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # 第三层
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # 最后一层轻微dropout

            # 输出层
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)
