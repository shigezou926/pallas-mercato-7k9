"""指定日の尼崎を予想する（展示タイム未取得なら展示前予想）。使い方: KDATA=... python3 predict_day.py 出走表.csv 20261127"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from common.data import load_venue, add_original
from common.features import build_base
from common import model as M, evaluate as E, tenji as T
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
for k in ["st", "shinnyu"]: D[k] = ext(D[k], np.full((n, 6), np.nan))   # 本番ST・進入は締切後の情報なので使わない
# 展示: 尼崎公式サイトの直前情報を今取得（展示タイム＋一周・まわり足）。失敗したらミラー(tenji.py)で展示タイムだけ。
# 取れない艇はNaN＝展示前扱い。
import requests
from venues.amagasaki import fetch_original as FO
TJ = np.full((n, 6), np.nan); IS = np.full((n, 6), np.nan); MW = np.full((n, 6), np.nan)
try:
    FO.fetch_day(day, requests.Session(), overwrite=True)
    o = pd.read_csv(C.ORIG_DIR / f"orig_{day}.csv", dtype={"登番": str}, encoding="utf-8-sig")
    for a, r in enumerate(rnos):
        g = card[card.レース == f"{r}R"].sort_values("枠")
        for x in o[o.レース == r].itertuples():
            if str(x.登番) != str(g.登録番号.values[x.枠 - 1]):
                continue   # 出走表と選手が違う（取り違え）→使わない
            TJ[a, x.枠 - 1] = x.展示タイム; IS[a, x.枠 - 1] = x.一周; MW[a, x.枠 - 1] = x.まわり足
    src = "尼崎サイト"
except Exception as e:
    print(f"[展示] 尼崎サイトから取得失敗 → ミラーで展示タイムのみ: {e}")
    try:
        TJ = T.to_array(T.fetch(C.JCD, day), rnos); src = "ミラー"
    except Exception as e2:
        print(f"[展示] ミラーも失敗 → 展示前予想: {e2}"); src = "なし"
print(f"[展示:{src}] 展示タイム {int(np.isfinite(TJ).all(1).sum())}/{n}R・一周 {int(np.isfinite(IS).any(1).sum())}/{n}R 取得（未取得のレースは展示前予想）")
D["tenji"] = ext(D["tenji"], TJ)
if C.USE_ORIGINAL:
    add_original(D, C.ORIG_DIR)   # 過去分（当日分は下で上書き）
    D["isshu"] = ext(D["isshu"][:N0], IS); D["mawari"] = ext(D["mawari"][:N0], MW)
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
