import torch.nn as nn


class DuelingQNet(nn.Module):
    def __init__(self, state_dim=40, action_dim=12, hidden_dim=256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 状态值分支（仅使用状态）
        self.value_stream = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出状态值 V(s)
        )

        # 动作优势分支（使用状态和动作）
        self.advantage_stream = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出动作优势 A(s, a)
        )

    def forward(self, x):
        # 提取状态部分（前 state_dim 位）
        state = x[:, :self.state_dim]

        # 计算状态值 V(s)
        V = self.value_stream(state)

        # 计算动作优势 A(s, a)
        A = self.advantage_stream(x)

        # 合并为 Q 值
        Q = V + (A - A.mean(dim=1, keepdim=True))  # 合并为Q值
        return Q