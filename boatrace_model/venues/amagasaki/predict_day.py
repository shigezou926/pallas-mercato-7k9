"""指定日の尼崎を予想する（展示タイム未取得なら展示前予想）。使い方: KDATA=... python3 predict_day.py 出走表.csv 20261127"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from common.data import load_venue
from common.features import build_base
from common import model as M, evaluate as E
from venues.amagasaki import config as C

card = pd.read_csv(sys.argv[1], dtype=str, encoding="utf-8-sig"); day = sys.argv[2]
card = card[(card.場コード == C.JCD) & (card.開催日 == day)]
D = load_venue(C.KDATA, C.JCD)
N0 = D["N"]; hd = np.array([k[0] for k in D["keys"]])
days = sorted(set(hd)); i_tune = int(np.argmax(hd >= days[C.TRAIN_DAYS]))
rnos = sorted(card.レース.str.replace("R", "").astype(int).unique()); n = len(rnos)
def ext(a, v): return np.concatenate([a, v])
D["N"] = N0 + n
for k in ["present"]: D[k] = ext(D[k], np.ones((n, 6), bool))
for k in ["tenji", "st", "shinnyu"]: D[k] = ext(D[k], np.full((n, 6), np.nan))   # 展示・STは未取得(使わない)
for k in ["win", "top2"]: D[k] = ext(D[k], np.zeros((n, 6)))
tb = np.empty((n, 6), object); mo = np.empty((n, 6), object)
for a, r in enumerate(rnos):
    g = card[card.レース == f"{r}R"].sort_values("枠")
    tb[a] = g.登録番号.values; mo[a] = g.モーターNo.astype(int).astype(str).values
D["toban"] = ext(D["toban"], tb); D["motor"] = ext(D["motor"], mo)
D["keys"] += [(day[2:], r) for r in rnos]; D["kimarite"] += [np.nan] * n; D["tansho"] += [{}] * n

F, base = build_base(D, i_tune)
w = M.fit(F, D["st"], D, np.arange(int(np.argmax(hd >= days[C.WARMUP_DAYS])), i_tune))
P, Xm, gap = M.predict(F, D, w, n_sim=C.N_SIM)
P = P[N0:]; gap = gap[N0:]
T = E.kimarite_table(D, range(i_tune))
rows = []
for a, r in enumerate(rnos):
    p = P[a]; top = int(p.argmax()); ratio = p / base["base_w"]; ratio[0] = 0; at = int(ratio.argmax())
    k = (p[:, None] * T).sum(0)
    lab = "本命" if p[top] >= 0.65 else ("攻め狙い(参考)" if (1 - p[0] >= 0.6 and p[at] >= 0.2 and ratio[at] >= 1.5) else "見送り")
    names = card[card.レース == f"{r}R"].sort_values("枠").氏名.str.replace("　", "").values
    rows.append(dict(R=r, 締切=card[card.レース == f"{r}R"].締切時刻.iloc[0], **{f"p{j+1}": round(float(p[j]), 3) for j in range(6)},
                     本命=top + 1, 攻め艇=at + 1, 一号艇危険度=round(1 - float(p[0]), 3), 判定=lab, 必要オッズ=round(1 / float(p[top]), 2),
                     逃げ=round(float(k[0]), 2), 差し=round(float(k[1]), 2), まくり=round(float(k[2]), 2), まくり差し=round(float(k[3]), 2)))
out = pd.DataFrame(rows); out.to_csv(C.OUT / f"today_{day}.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); print(out.to_string(index=False))
