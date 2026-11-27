"""モデル＝条件付きロジット(1着確率) + スタート(ST)のモンテカルロ。

学習: 実際のSTで「その艇が1着になる」確率を学習。
予測: 本番STは分からないので、選手の過去STから多数回サンプリング(のぞき・凹みの隊形を再現)し、
      各試行の1着確率を平均する → 1〜6号艇の単勝確率。
"""
import numpy as np
from .features import build_X, P


def softmax_masked(s, mask):
    s = np.where(mask, s, -np.inf)
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(-1, keepdims=True)


def fit(F, st_real, D, idx, lam=3.0, iters=40):
    st = np.where(D["present"], st_real, np.nan)
    # 欠測STは予測値で補う
    st = np.where(np.isnan(st) & D["present"], F["mu"], st)
    X = build_X({k: v[idx] for k, v in F.items()}, st[idx][:, None, :], D["present"][idx])[:, 0]
    M = D["present"][idx]; y = D["win"][idx].argmax(1)
    w = np.zeros(P); n = len(idx)
    for _ in range(iters):
        p = softmax_masked(X @ w, M)
        xbar = np.einsum("nj,njp->np", p, X)
        grad = (X[np.arange(n), y] - xbar).sum(0) - lam * w
        d = X - xbar[:, None, :]
        H = np.einsum("nj,njp,njq->pq", p, d, d) + lam * np.eye(P)
        step = np.linalg.solve(H, grad)
        w += step
        if np.abs(step).max() < 1e-7:
            break
    return w


def predict(F, D, w, n_sim=400, seed=0, chunk=150):
    rng = np.random.default_rng(seed)
    N = D["N"]
    out = np.zeros((N, 6)); Xm = np.zeros((N, 6, P)); gap = np.zeros((N, 6))
    for a in range(0, N, chunk):
        sl = slice(a, min(N, a + chunk))
        mu, sd = F["mu"][sl], F["sd"][sl]
        st = mu[:, None, :] + sd[:, None, :] * rng.standard_normal((mu.shape[0], n_sim, 6))
        X = build_X({k: v[sl] for k, v in F.items()}, st, D["present"][sl])
        p = softmax_masked(X @ w, D["present"][sl][:, None, :])
        out[sl] = p.mean(1); Xm[sl] = X.mean(1)
        gap[sl] = (X[..., 12] + X[..., 13]).mean(1) / 10        # 内より早い度合い(秒)
    return out, Xm, gap
