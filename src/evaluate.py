# 테스트 100개 엔진의 마지막 시점 RUL 예측 평가

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import RulLSTM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/processed')
    parser.add_argument('--model', default='results/best_model.pt')
    parser.add_argument('--out', default='results')
    args = parser.parse_args()

    d = np.load(os.path.join(args.data, 'test.npz'))
    X, y = d['X'], d['y']
    with open(os.path.join(args.data, 'meta.json')) as f:
        scale = float(json.load(f)['clip'])  # 모델이 0~1로 예측하므로 복원용

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RulLSTM(n_features=X.shape[-1]).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    with torch.no_grad():
        pred = model(torch.from_numpy(X).to(device)).cpu().numpy() * scale

    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    mae = float(np.mean(np.abs(pred - y)))
    print(f'test RMSE: {rmse:.2f} cycles')
    print(f'test MAE : {mae:.2f} cycles')

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'test_metrics.txt'), 'w') as f:
        f.write(f'test RMSE: {rmse:.2f}\ntest MAE: {mae:.2f}\n')

    # 실제 vs 예측 산점도
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y, pred, s=18, alpha=0.7)
    lim = max(y.max(), pred.max()) + 10
    ax.plot([0, lim], [0, lim], 'r--', lw=1)
    ax.set_xlabel('true RUL (cycles)')
    ax.set_ylabel('predicted RUL (cycles)')
    ax.set_title(f'test units (n={len(y)}), RMSE={rmse:.1f}')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'pred_vs_true.png'), dpi=150)

    # RUL이 작은(고장 임박) 엔진일수록 정확한지 확인
    order = np.argsort(y)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y[order], label='true')
    ax.plot(pred[order], label='pred', alpha=0.8)
    ax.set_xlabel('test unit (sorted by true RUL)')
    ax.set_ylabel('RUL (cycles)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'per_unit.png'), dpi=150)


if __name__ == '__main__':
    main()
