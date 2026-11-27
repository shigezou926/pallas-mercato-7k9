import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from common.data import load_venue, add_original
from common.features import build_base, FEATURE_NAMES
from common import model as M
from common import evaluate as E
from venues.amagasaki import config as C

D = load_venue(C.KDATA, C.JCD)
if C.USE_ORIGINAL:
    print("[オリジナル展示]", add_original(D, C.ORIG_DIR),
          "取得済みレース率", round(float(np.isfinite(D["isshu"]).any(1).mean()), 3))
N = D["N"]; hd = np.array([k[0] for k in D["keys"]])
days = sorted(set(hd))
d_tune, d_hold = days[C.TRAIN_DAYS], days[C.TRAIN_DAYS + C.TUNE_DAYS]
i_tune = int(np.argmax(hd >= d_tune)); i_hold = int(np.argmax(hd >= d_hold))
split = np.where(np.arange(N) < i_tune, "学習", np.where(np.arange(N) < i_hold, "調整", "確認"))
split = split.astype(object); split[hd >= C.NEW_FROM] = "新確認"
print(f"レース{N} 開催日{len(days)} 除外{D['dropped']}  学習<{d_tune} 調整<{d_hold} 確認以降")

F, base = build_base(D, i_tune)
warm = int(np.argmax(hd >= days[C.WARMUP_DAYS]))
fit_idx = np.arange(warm, i_tune)
w = M.fit(F, D["st"], D, fit_idx)
P, Xm, gap = M.predict(F, D, w, n_sim=C.N_SIM)
T = E.kimarite_table(D, range(i_tune))

win = D["win"]; pres = D["present"]
bw = base["base_w"]
P_course = pres * bw; P_course = P_course / P_course.sum(1, keepdims=True)
P_one = np.zeros((N, 6)); P_one[:, 0] = 1                       # 常に1号艇(確率としては0/1)
P_one_s = 0.9 * P_one + 0.1 / 6                                  # Brier用に少し丸めた版は使わず、参考のみ

def idx_of(name): return np.where(split == name)[0]
tune, hold = idx_of("調整"), idx_of("新確認")
best_roi, th = E.grid(D, tune, P[tune], gap[tune], bw)
th["base_w"] = bw
lab, tgt, att, danger = E.judge(P, gap, th)
pnl, bt = E.bet_pnl(D, np.arange(N), tgt)
tgt1 = np.zeros(N, int); pnl1, bt1 = E.bet_pnl(D, np.arange(N), tgt1)
hit = np.array([win[i, t] if t >= 0 else 0 for i, t in enumerate(tgt)], float)
hit1 = win[:, 0]
top = P.argmax(1)
tgt_top = top.copy(); pnl_top, bt_top = E.bet_pnl(D, np.arange(N), tgt_top)

# --- 期間別 ---
rows = []
month = np.array([h[:4] for h in hd])
def block(name, ix):
    r = dict(区分=name, レース=len(ix),
             Brier_モデル=E.brier(P[ix], win[ix]), Brier_コース平均=E.brier(P_course[ix], win[ix]),
             Brier_常に1号艇=E.brier(P_one[ix], win[ix]),
             LogLoss_モデル=E.logloss(P[ix], win[ix]), LogLoss_コース平均=E.logloss(P_course[ix], win[ix]),
             最有力艇の1着率=float(win[ix, top[ix]].mean()), **{"1号艇1着率": float(win[ix, 0].mean())})
    s = E.summary(pnl[ix], bt[ix], hit[ix])
    r.update({f"判定_{k}": v for k, v in s.items()})
    s1 = E.summary(pnl1[ix], bt1[ix], hit1[ix]); r.update({f"常に1号艇_{k}": v for k, v in s1.items()})
    st = E.summary(pnl_top[ix], bt_top[ix], win[ix, top[ix]]); r.update({f"常に最有力艇_{k}": v for k, v in st.items()})
    sa = bt[ix] & (tgt[ix] > 0)
    r.update({"1号艇以外_購入": int(sa.sum()),
              "1号艇以外_的中率": float(hit[ix][sa].mean()) if sa.any() else np.nan,
              "1号艇以外_回収率": float((pnl[ix][sa].sum() + 100 * sa.sum()) / (100 * sa.sum())) if sa.any() else np.nan})
    return r
for s in ["学習", "調整", "確認", "新確認"]:
    rows.append(block(s, idx_of(s)))
for m in sorted(set(month)):
    ix = np.where(month == m)[0]
    rows.append(block(f"月 20{m[:2]}-{m[2:]}（{split[ix[0]]}）", ix))
R = pd.DataFrame(rows); R.to_csv(C.OUT / "period_summary.csv", index=False, encoding="utf-8-sig")

# --- 予測出力 ---
def names(idx_list): return "".join(str(i + 1) for i in idx_list)
recs = []
wl = w
for i in range(N):
    p = P[i]; f = int(top[i]); a = int(att[i])
    k = (p[:, None] * T).sum(0)
    # 展開をもらう艇: 攻め艇の両隣のうち確率が高い方
    nb = [j for j in (a - 1, a + 1) if 0 <= j < 6 and j != 0 and pres[i, j]] or [f]
    rcv = max(nb, key=lambda j: p[j])
    contrib = wl[None, :] * (Xm[i, f] - Xm[i].mean(0))[None, :]
    c = contrib[0].copy(); c[:5] = 0; order = np.argsort(-c)[:3]
    reason = "・".join(f"{FEATURE_NAMES[o]}" for o in order if c[o] > 0.02)
    c1 = wl * (Xm[i, 0] - Xm[i].mean(0)); c1[:5] = 0; o1 = np.argsort(c1)[:2]
    r1 = "・".join(FEATURE_NAMES[o] for o in o1 if c1[o] < -0.02) or "目立つ不利材料なし"
    recs.append(dict(日付="20" + hd[i], R=D["keys"][i][1], 区分=split[i],
                     **{f"p{j+1}": round(float(p[j]), 4) for j in range(6)},
                     本命=f + 1, 攻め艇=a + 1, 展開をもらう艇=rcv + 1, 一号艇危険度=round(float(danger[i]), 3),
                     判定=lab[i], 買い艇=(int(tgt[i]) + 1 if tgt[i] >= 0 else ""),
                     必要オッズ=round(1 / float(p[f]), 2),
                     **{f"展開_{n}": round(float(k[q]), 3) for q, n in enumerate(E.KIM)},
                     根拠=f"{f+1}号艇は{("最内の枠有利＋" if f == 0 else "") + (reason or "枠の差のみ")}が効く。1号艇の不安: {r1}。攻め艇は{a+1}号艇(内との差{gap[i,a]*1000:+.0f}ms・+は自艇が早い)",
                     実際の勝ち艇=int(win[i].argmax()) + 1))
pd.DataFrame(recs).to_csv(C.OUT / "predictions.csv", index=False, encoding="utf-8-sig")
json.dump(dict(係数={n: round(float(v), 4) for n, v in zip(FEATURE_NAMES, w)}, 閾値={k: (v if k != "base_w" else list(map(float, v))) for k, v in th.items()},
               調整期間ROI=best_roi, 分割=dict(学習=f"〜{d_tune}未満", 調整=f"{d_tune}〜{d_hold}未満", 確認=f"{d_hold}〜")),
          open(C.OUT / "model.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# --- 校正 ---
cal = pd.DataFrame(dict(p=P[hold][pres[hold]], y=win[hold][pres[hold]]))
cal["bin"] = pd.cut(cal.p, [0, .05, .1, .15, .2, .3, .4, .5, .6, .7, 1])
print("\n[校正:新確認期間] 予測確率の帯ごとの実際の1着率"); print(cal.groupby("bin", observed=True).agg(件数=("y", "size"), 予測平均=("p", "mean"), 実際=("y", "mean")).round(3).to_string())
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
show = R[R.区分.isin(["学習", "調整", "確認", "新確認"])].set_index("区分").T
print("\n[期間別]"); print(show.to_string(float_format=lambda x: f"{x:.4f}"))
print("\n[月別]"); print(R[R.区分.str.startswith("月")][["区分", "レース", "Brier_モデル", "Brier_コース平均", "最有力艇の1着率", "1号艇1着率", "判定_購入", "判定_的中率", "判定_回収率", "常に1号艇_回収率"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\n閾値", {k: v for k, v in th.items() if k != "base_w"}, "調整期間ROI", round(best_roi, 3))
print("係数", {n: round(float(v), 3) for n, v in zip(FEATURE_NAMES, w)})
print("判定内訳(確認):", pd.Series(lab[hold]).value_counts().to_dict())

# --- 参考: 1号艇以外を買う「攻め狙い」を固定閾値で見る(調整に使わない・採否の根拠にしない) ---
print("\n[参考:攻め狙い(固定閾値)]  区分別  購入/的中率/回収率 | 攻め艇の平均予測確率")
for t_danger, t_att, t_ratio in [(0.5, 0.20, 1.5), (0.6, 0.20, 1.5), (0.6, 0.25, 1.5), (0.7, 0.20, 1.0)]:
    th2 = dict(t_hon=1.01, t_danger=t_danger, t_att=t_att, t_ratio=t_ratio, base_w=bw)
    _, tg2, _, _ = E.judge(P, gap, th2)
    pn2, b2 = E.bet_pnl(D, np.arange(N), tg2)
    h2 = np.array([win[i, t] if t >= 0 else 0 for i, t in enumerate(tg2)], float)
    line = []
    for sname in ["学習", "調整", "確認", "新確認"]:
        ix = idx_of(sname); sm = E.summary(pn2[ix], b2[ix], h2[ix])
        line.append(f"{sname}:{sm['購入']}点 的中{sm['的中率']:.3f} 回収{sm['回収率']:.3f}" if sm["購入"] else f"{sname}:0点")
    pm = np.mean([P[i, t] for i, t in enumerate(tg2) if t >= 0]) if (tg2 >= 0).any() else float("nan")
    print(f" 危険度>={t_danger} 攻め確率>={t_att} 比>={t_ratio}: " + " | ".join(line) + f" | 平均p={pm:.3f}")
