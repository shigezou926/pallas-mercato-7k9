"""特徴量：各レースの時点で『それ以前に終わったレース』だけから作る。

使う情報（すべて締切前に分かる）:
  展示タイム(当該レース)、選手の過去成績(当地・過去分のみ)、モーターの直近成績、選手の過去ST(コース補正)
使わない情報: 当該レースの本番ST・着順・進入・風/波(レース後の実況値で、締切前の履歴がないため)
コースは「枠なり」で固定（実際の進入は締切後にしか分からない）。
"""
from collections import deque
import numpy as np


def build_base(D, n_fit, K_skill=15, K_motor=10, K_st=8, motor_win=40):
    N = D["N"]
    pres, win, top2, st = D["present"], D["win"], D["top2"], D["st"]
    # 以下の定数は学習期間(先頭n_fitレース)だけから作る → 調整/確認期間の情報は入らない
    base_w = (win[:n_fit] * pres[:n_fit]).sum(0) / pres[:n_fit].sum(0)
    base_t2 = (top2[:n_fit] * pres[:n_fit]).sum(0) / pres[:n_fit].sum(0)
    off = np.nanmean(st[:n_fit], axis=0)                       # コース別の平均ST
    gsd = np.nanstd(st[:n_fit] - off)
    F = {k: np.zeros((N, 6)) for k in ["tenji_rel", "skill_w", "skill_t2", "motor_t2", "mu", "sd"]}
    racer, motor = {}, {}
    for i in range(N):
        pr = pres[i]
        tj = D["tenji"][i]
        tm = np.nanmean(np.where(pr, tj, np.nan)) if np.isfinite(tj[pr]).any() else 0.0
        for j in range(6):
            if not pr[j]:
                continue
            s = racer.get(D["toban"][i, j], [0, 0.0, 0.0, 0, 0.0, 0.0])
            F["skill_w"][i, j] = s[1] / (s[0] + K_skill)
            F["skill_t2"][i, j] = s[2] / (s[0] + K_skill)
            F["mu"][i, j] = off[j] + s[4] / (s[3] + K_st)
            F["sd"][i, j] = np.sqrt((s[5] + K_st * gsd ** 2) / (s[3] + K_st))
            m = motor.get(D["motor"][i, j])
            F["motor_t2"][i, j] = (sum(m) / (len(m) + K_motor)) if m else 0.0
            F["tenji_rel"][i, j] = (tj[j] - tm) if np.isfinite(tj[j]) else 0.0
        # --- 結果が確定したあとで状態を更新（次のレース以降にだけ効く）---
        for j in range(6):
            if not pr[j]:
                continue
            k = D["toban"][i, j]
            s = racer.setdefault(k, [0, 0.0, 0.0, 0, 0.0, 0.0])
            s[0] += 1
            s[1] += win[i, j] - base_w[j]
            s[2] += top2[i, j] - base_t2[j]
            if np.isfinite(st[i, j]):
                res = st[i, j] - off[j]
                s[3] += 1; s[4] += res; s[5] += res * res
            motor.setdefault(D["motor"][i, j], deque(maxlen=motor_win)).append(top2[i, j] - base_t2[j])
    return F, dict(base_w=base_w, base_t2=base_t2, off=off, gsd=gsd)


FEATURE_NAMES = ["枠2", "枠3", "枠4", "枠5", "枠6", "展示タイム(速いほど)", "選手勝ち力", "選手連対力",
                 "モーター", "ST(1枠)", "ST(2-3枠)", "ST(4-6枠)", "内より早い(2-3枠)",
                 "内より早い(4-6枠)", "外が遅い(1枠)", "外が遅い(2-3枠)"]
P = len(FEATURE_NAMES)


def build_X(F, st, present):
    """st: (N,D,6)  F: (N,6) ... → X (N,D,6,P)。NaNは0。"""
    N, Dn, _ = st.shape
    stm = np.where(present[:, None, :], st, np.nan)
    with np.errstate(all="ignore"):
        rel = stm - np.nanmean(stm, axis=2, keepdims=True)
        gin = np.zeros_like(stm); gout = np.zeros_like(stm)
        for j in range(6):
            if j > 0:
                gin[..., j] = np.nanmean(stm[..., max(0, j - 2):j], axis=2) - stm[..., j]
            if j < 5:
                gout[..., j] = np.nanmean(stm[..., j + 1:j + 3], axis=2) - stm[..., j]
    rel, gin, gout = [np.nan_to_num(a) * 10 for a in (rel, gin, gout)]   # 秒→0.1秒単位
    X = np.zeros((N, Dn, 6, P))
    for j in range(1, 6):
        X[:, :, j, j - 1] = 1.0
    X[..., 5] = -F["tenji_rel"][:, None, :] * 10
    X[..., 6] = F["skill_w"][:, None, :] * 10
    X[..., 7] = F["skill_t2"][:, None, :] * 10
    X[..., 8] = F["motor_t2"][:, None, :] * 10
    c = np.arange(6)
    g1, g23, g456 = (c == 0), (c == 1) | (c == 2), c >= 3
    X[..., 9] = rel * g1; X[..., 10] = rel * g23; X[..., 11] = rel * g456
    X[..., 12] = gin * g23; X[..., 13] = gin * g456
    X[..., 14] = gout * g1; X[..., 15] = gout * g23
    X *= present[:, None, :, None]
    return X
