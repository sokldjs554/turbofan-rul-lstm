# C-MAPSS FD001 txt -> 학습용 npz 변환
# train은 run-to-failure(고장까지 전체), test는 중간에 잘려 있고
# 잘린 시점의 실제 RUL이 RUL_FD001.txt에 있음

import argparse
import json
import os

import numpy as np
import pandas as pd

COLS = (['unit', 'cycle', 'set1', 'set2', 'set3']
        + [f's{i}' for i in range(1, 22)])

# FD001은 운전조건이 하나라 값이 거의 안 변하는 센서가 있음
# (EDA에서 std 확인하고 제외한 목록)
DROP_SENSORS = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']
FEATURES = [c for c in [f's{i}' for i in range(1, 22)] if c not in DROP_SENSORS]


def load_txt(path):
    df = pd.read_csv(path, sep=r'\s+', header=None)
    df.columns = COLS
    return df


def make_windows(df, features, window, clip):
    # unit별로 슬라이딩 윈도우. 마지막 cycle 기준 RUL이 라벨
    Xs, ys, units = [], [], []
    for uid, g in df.groupby('unit'):
        arr = g[features].to_numpy(dtype=np.float32)
        max_cycle = g['cycle'].max()
        if len(arr) < window:
            # 짧은 unit은 첫 행을 반복해서 패딩 (FD001엔 거의 없지만 안전장치)
            pad = np.repeat(arr[:1], window - len(arr), axis=0)
            arr = np.vstack([pad, arr])
        for end in range(window, len(arr) + 1):
            Xs.append(arr[end - window:end])
            rul = max_cycle - g['cycle'].iloc[min(end, len(g)) - 1]
            ys.append(min(rul, clip))
            units.append(uid)
    return (np.stack(Xs), np.array(ys, dtype=np.float32),
            np.array(units, dtype=np.int64))


def last_window(df, features, window):
    Xs = []
    for _, g in df.groupby('unit'):
        arr = g[features].to_numpy(dtype=np.float32)
        if len(arr) < window:
            pad = np.repeat(arr[:1], window - len(arr), axis=0)
            arr = np.vstack([pad, arr])
        Xs.append(arr[-window:])
    return np.stack(Xs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--out', default='data/processed')
    parser.add_argument('--window', type=int, default=30)
    parser.add_argument('--clip', type=int, default=125,
                        help='RUL 상한. 초반 구간은 고장 징후가 없어서 '
                             '큰 RUL 값이 라벨 노이즈처럼 작동함')
    parser.add_argument('--val-units', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train = load_txt(os.path.join(args.data_dir, 'train_FD001.txt'))
    test = load_txt(os.path.join(args.data_dir, 'test_FD001.txt'))
    rul_true = pd.read_csv(os.path.join(args.data_dir, 'RUL_FD001.txt'),
                           sep=r'\s+', header=None)[0].to_numpy(np.float32)

    # 정규화 파라미터는 train에서만 산출 (test 정보 누설 방지)
    f_min = train[FEATURES].min().to_numpy(np.float32)
    f_max = train[FEATURES].max().to_numpy(np.float32)
    rng_ = np.where(f_max - f_min == 0, 1, f_max - f_min)
    for df in (train, test):
        df[FEATURES] = (df[FEATURES] - f_min) / rng_

    X, y, units = make_windows(train, FEATURES, args.window, args.clip)

    # 같은 엔진의 윈도우가 train/val 양쪽에 들어가면 안 되니까 unit 단위로 분할
    rng = np.random.default_rng(args.seed)
    all_units = np.unique(units)
    val_ids = rng.choice(all_units, args.val_units, replace=False)
    val_mask = np.isin(units, val_ids)

    X_te = last_window(test, FEATURES, args.window)
    y_te = np.minimum(rul_true, args.clip)

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, 'train.npz'),
                        X=X[~val_mask], y=y[~val_mask])
    np.savez_compressed(os.path.join(args.out, 'val.npz'),
                        X=X[val_mask], y=y[val_mask])
    np.savez_compressed(os.path.join(args.out, 'test.npz'), X=X_te, y=y_te)

    meta = {'features': FEATURES, 'window': args.window, 'clip': args.clip,
            'f_min': f_min.tolist(), 'f_max': f_max.tolist(),
            'val_units': sorted(int(u) for u in val_ids)}
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'train {sum(~val_mask)} / val {sum(val_mask)} / test {len(X_te)}')
    print('저장 완료:', args.out)


if __name__ == '__main__':
    main()
