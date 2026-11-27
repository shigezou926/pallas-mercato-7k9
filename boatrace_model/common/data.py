"""取込：kdata（読み取り専用）から1場ぶんを (レース×6枠) の配列にする。元データは変更しない。"""
from pathlib import Path
import numpy as np
import pandas as pd


def parse_st(s):
    """ST文字列→(値, フラグ)。F0.03=フライング(早い→負値で保持)、L=出遅れ(0.30扱い)。"""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    s = str(s)
    if s.startswith("F"):
        return -float(s[1:])
    if s == "L":
        return 0.30
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_venue(kdata: Path, jcd: str):
    e = pd.read_csv(kdata / "entriesFull.csv", dtype=str)
    e = e[e.jcd == jcd].copy()
    r = pd.read_csv(kdata / "racesAll.csv", dtype=str)
    r = r[r.jcd == jcd].set_index(["hd", "rno"])
    pays = [pd.read_csv(f, dtype=str) for f in sorted(kdata.glob("payouts2*.csv"))]
    p = pd.concat(pays)
    p = p[(p.jcd == jcd) & (p.shiki == "単勝")]
    tansho = {}
    for hd, rno, combo, pay in zip(p.hd, p.rno, p.combo, p.payout):
        if not (str(combo).isdigit() and str(pay).isdigit()):      # 特払い等の非数値行は払戻なし(0)扱い
            continue
        tansho.setdefault((hd, int(rno)), {})[int(combo)] = int(pay)

    e["rno"] = e.rno.astype(int)
    e["waku"] = e.waku.astype(int)
    e = e.sort_values(["hd", "rno", "waku"])
    keys, rows, dropped = [], [], {"同着/勝者なし": 0, "6枠そろわず": 0}
    for (hd, rno), g in e.groupby(["hd", "rno"], sort=True):
        if len(g) != 6 or sorted(g.waku) != [1, 2, 3, 4, 5, 6]:
            dropped["6枠そろわず"] += 1
            continue
        if (g.chaku == "01").sum() != 1:
            dropped["同着/勝者なし"] += 1
            continue
        keys.append((hd, rno))
        rows.append(g)
    N = len(keys)
    A = lambda: np.full((N, 6), np.nan)
    D = dict(N=N, keys=keys, present=np.zeros((N, 6), bool), tenji=A(), st=A(),
             toban=np.empty((N, 6), object), motor=np.empty((N, 6), object),
             win=np.zeros((N, 6)), top2=np.zeros((N, 6)), shinnyu=A(),
             kimarite=[], tansho=[], dropped=dropped)
    for i, g in enumerate(rows):
        g = g.sort_values("waku")
        D["present"][i] = g.shinnyu.notna().values
        D["tenji"][i] = pd.to_numeric(g.tenjiT).values
        D["st"][i] = [parse_st(x) for x in g.st]
        D["toban"][i] = g.toban.values
        D["motor"][i] = g.motorNo.values
        D["win"][i] = (g.chaku == "01").values
        D["top2"][i] = g.chaku.isin(["01", "02"]).values
        D["shinnyu"][i] = pd.to_numeric(g.shinnyu).values
        k = keys[i]
        kk = (k[0], str(k[1]))
        D["kimarite"].append(r.kimarite.get(kk, np.nan) if kk in r.index else np.nan)
        D["tansho"].append(tansho.get(k, {}))
    return D
