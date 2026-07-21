# C-MAPSS 형식의 합성 데이터 생성 (smoke test용)
# 진짜 데이터 없이 전처리~학습 파이프라인 동작 확인 용도

import argparse
import os

import numpy as np


def make_unit(uid, n_cycles, rng, n_sensors=21):
    rows = []
    # 열화 곡선: 초반엔 평탄하다가 후반에 지수적으로 변함
    t = np.arange(n_cycles) / n_cycles
    degrade = np.exp(3 * (t - 1))
    for c in range(n_cycles):
        sensors = []
        for s in range(n_sensors):
            base = 500 + s * 20
            if s in (0, 4, 5, 9, 15, 17, 18):  # 상수 센서 흉내 (s1,s5,...)
                sensors.append(base)
            else:
                drift = (1 if s % 2 else -1) * 30 * degrade[c]
                sensors.append(base + drift + rng.normal(0, 1.5))
        rows.append([uid, c + 1, rng.normal(0, .002), rng.normal(0, .0003),
                     100.0] + sensors)
    return rows


def write_txt(path, rows):
    with open(path, 'w') as f:
        for r in rows:
            f.write(' '.join(f'{v:.4f}' if isinstance(v, float) else str(v)
                             for v in r) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data')
    parser.add_argument('--units', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    train_rows = []
    for u in range(1, args.units + 1):
        train_rows += make_unit(u, int(rng.integers(120, 250)), rng)
    write_txt(os.path.join(args.out, 'train_FD001.txt'), train_rows)

    test_rows, ruls = [], []
    for u in range(1, args.units + 1):
        full = int(rng.integers(120, 250))
        cut = int(rng.integers(40, full - 10))
        rows = make_unit(u, full, rng)[:cut]
        test_rows += rows
        ruls.append(full - cut)
    write_txt(os.path.join(args.out, 'test_FD001.txt'), test_rows)
    with open(os.path.join(args.out, 'RUL_FD001.txt'), 'w') as f:
        f.write('\n'.join(str(r) for r in ruls) + '\n')

    print(f'합성 데이터 저장 -> {args.out} (units={args.units})')


if __name__ == '__main__':
    main()
