import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import RulLSTM


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(path):
    d = np.load(path)
    return TensorDataset(torch.from_numpy(d['X']), torch.from_numpy(d['y']))


@torch.no_grad()
def rmse(model, loader, device, scale):
    model.eval()
    se, n = 0.0, 0
    for x, y in loader:
        pred = model(x.to(device)) * scale  # 0~1 예측을 다시 cycle 단위로
        se += ((pred.cpu() - y) ** 2).sum().item()
        n += len(y)
    return (se / n) ** 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/processed')
    parser.add_argument('--out', default='results')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)

    # 라벨을 0~1로 스케일해서 학습.
    # 처음에 RUL(0~125)을 그대로 회귀시켰더니 초반 gradient가 너무 커서
    # head의 ReLU가 죽고 모델이 평균값만 찍는 문제가 있었음 (RMSE 37에서 정체).
    # 타깃을 clip 값으로 나눠주니 정상적으로 학습됨.
    with open(os.path.join(args.data, 'meta.json')) as f:
        scale = float(json.load(f)['clip'])

    train_ds = load_split(os.path.join(args.data, 'train.npz'))
    val_ds = load_split(os.path.join(args.data, 'val.npz'))
    print(f'train {len(train_ds)} / val {len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)

    n_features = train_ds.tensors[0].shape[-1]
    model = RulLSTM(n_features=n_features).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    epochs = 1 if args.smoke else args.epochs
    os.makedirs(args.out, exist_ok=True)
    history = {'train_loss': [], 'val_rmse': []}
    best = float('inf')

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y / scale)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        val_rmse = rmse(model, val_loader, device, scale)
        history['train_loss'].append(float(np.mean(losses)))
        history['val_rmse'].append(float(val_rmse))

        mark = ''
        if val_rmse < best:
            best = val_rmse
            torch.save(model.state_dict(),
                       os.path.join(args.out, 'best_model.pt'))
            mark = ' *'
        print(f'[{epoch+1:2d}/{epochs}] loss {np.mean(losses):.4f} | '
              f'val RMSE {val_rmse:.2f} | {time.time()-t0:.0f}s{mark}')

    with open(os.path.join(args.out, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history['train_loss'])
    ax[0].set_title('train loss (MSE)')
    ax[0].set_xlabel('epoch')
    ax[1].plot(history['val_rmse'])
    ax[1].set_title('val RMSE (cycles)')
    ax[1].set_xlabel('epoch')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'training_curve.png'), dpi=150)

    print(f'best val RMSE: {best:.2f}')


if __name__ == '__main__':
    main()
