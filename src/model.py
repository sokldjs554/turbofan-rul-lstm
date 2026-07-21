import torch.nn as nn


class RulLSTM(nn.Module):
    """(batch, window, features) -> RUL 스칼라 예측.

    GRU랑 비교했을 때 차이가 거의 없어서 (val RMSE 0.3 정도)
    더 흔히 쓰이는 LSTM으로 확정. 층수는 2가 1보다 확실히 나았고
    3부터는 이득이 없었음.
    """

    def __init__(self, n_features=14, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)  # 마지막 타임스텝만 사용
