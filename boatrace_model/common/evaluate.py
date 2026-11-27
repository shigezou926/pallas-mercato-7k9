"""判定・検証。購入は単勝100円の均一買い、払戻は確定単勝配当（直前オッズは手元にない）。"""
import numpy as np
import pandas as pd

KIM = ["逃げ", "差し", "まくり", "まくり差し", "抜き", "恵まれ"]


def brier(p, win):          # 6艇ぶんの二乗誤差の和を、レース平均
    return float(((p - win) ** 2).sum(1).mean())


def logloss(p, win):
    return float(-np.log(np.clip((p * win).sum(1), 1e-9, 1)).mean())


def kimarite_table(D, idx):
    T = np.ones((6, len(KIM)))      # ラプラス平滑
    for i in idx:
        k = D["kimarite"][i]
        if k in KIM:
            T[D["win"][i].argmax(), KIM.index(k)] += 1
    return T / T.sum(1, keepdims=True)


def judge(p, gap, th):
    """本命 / 攻め狙い / 見送り と、買う艇(0始まり。見送りは-1)を返す。"""
    N = len(p); lab = np.array(["見送り"] * N, dtype=object); tgt = -np.ones(N, int)
    top = p.argmax(1); ptop = p.max(1)
    danger = 1 - p[:, 0]
    base = th["base_w"]
    ratio = p / base
    ratio[:, 0] = 0
    att = ratio.argmax(1); patt = p[np.arange(N), att]
    hon = ptop >= th["t_hon"]
    ata = (~hon) & (danger >= th["t_danger"]) & (patt >= th["t_att"]) & (ratio[np.arange(N), att] >= th["t_ratio"])
    lab[hon] = "本命"; tgt[hon] = top[hon]
    lab[ata] = "攻め狙い"; tgt[ata] = att[ata]
    return lab, tgt, att, danger


def bet_pnl(D, idx, tgt):
    """各レース: 買わない=0 / 買う=払戻-100。"""
    pnl = np.zeros(len(idx)); bought = tgt >= 0
    for n, (i, t) in enumerate(zip(idx, tgt)):
        if t < 0:
            continue
        pnl[n] = D["tansho"][i].get(int(t) + 1, 0) - 100
    return pnl, bought


def summary(pnl, bought, hit):
    b = int(bought.sum())
    if b == 0:
        return dict(購入=0, 的中率=np.nan, 回収率=np.nan, 損益=0, 最大の落ち込み=0)
    cum = np.cumsum(pnl)
    dd = float((np.maximum.accumulate(np.r_[0, cum]) - np.r_[0, cum]).max())
    return dict(購入=b, 的中率=float(hit[bought].mean()), 回収率=float((pnl[bought].sum() + 100 * b) / (100 * b)),
                損益=int(pnl.sum()), 最大の落ち込み=int(dd))


def grid(D, idx, p, gap, base_w, min_bets=40):
    best = None
    for t_hon in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 1.01]:
        for t_danger in [0.5, 0.6, 0.65, 0.7, 0.75, 1.01]:
            for t_att in [0.15, 0.2, 0.25, 0.3]:
                for t_ratio in [1.0, 1.5, 2.0]:
                    th = dict(t_hon=t_hon, t_danger=t_danger, t_att=t_att, t_ratio=t_ratio, base_w=base_w)
                    lab, tgt, _, _ = judge(p, gap, th)
                    pnl, bt = bet_pnl(D, idx, tgt)
                    if bt.sum() < min_bets:
                        continue
                    roi = (pnl[bt].sum() + 100 * bt.sum()) / (100 * bt.sum())
                    if best is None or roi > best[0]:
                        best = (roi, th)
    return best
