import torch.nn as nn

# class BalancedClassifier(nn.Module):
#     def __init__(self, input_dim, num_classes): 
#         super().__init__()
#         self.net = nn.Sequential(
#             # 第一层
#             nn.Linear(input_dim, 256),
#             nn.BatchNorm1d(256),  # 添加Batch Normalization
#             nn.ReLU(),
#             nn.Dropout(0.4),  # 适中的dropout

#             # 第二层
#             nn.Linear(256, 128),
#             nn.BatchNorm1d(128),  # 添加Batch Normalization
#             nn.ReLU(),
#             nn.Dropout(0.3),

#             # 第三层
#             nn.Linear(128, 64),
#             nn.BatchNorm1d(64),  # 添加Batch Normalization
#             nn.ReLU(),
#             nn.Dropout(0.2), 
            
#             #第四层
#             nn.Linear(64, 32),
#             nn.BatchNorm1d(32),  # 添加Batch Normalization
#             nn.ReLU(),
#             nn.Dropout(0.1),

#             # 输出层
#             nn.Linear(32, num_classes)
#         )

#     def forward(self, x):
#         return self.net(x)

import torch.nn as nn

class BalancedClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,32),
            nn.SiLU(),          # 或 nn.ReLU()
            nn.Dropout(0.15),

            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Dropout(0.10),

            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return self.net(x)
