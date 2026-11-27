"""尼崎のオリジナル展示（一周・まわり足）＋展示タイム・体重・チルト・水面気象を取得する。

取得元: ボートレース尼崎 公式サイト（スマホ版「直前情報」）
  https://www.boatrace-amagasaki.jp/sp/ajax/ajax_yosou.php?targetday=YYYYMMDD&race=R&req=cyokuzen&run=0
  ※サイト自身のページが読み込む部品。X-Requested-With ヘッダが無いと空を返す。
一周・まわり足は尼崎の独自計測。計測できなかった場合は「-」→ 空欄のまま（補完しない）。

使い方:
  python fetch_original.py 20261127               # 1日分（当日は締切前に何度でも取り直してよい）
  python fetch_original.py 20250701 20261126      # 期間をまとめて（保存済みの日は飛ばす）
保存先: venues/amagasaki/original/orig_YYYYMMDD.csv（.gitignore 済み。公開リポジトリには上げない）
"""
import datetime
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

URL = "https://www.boatrace-amagasaki.jp/sp/ajax/ajax_yosou.php?targetday={d}&race={r}&req=cyokuzen&run=0"
HEAD = {"User-Agent": "Mozilla/5.0 (boatrace-model; personal research)", "X-Requested-With": "XMLHttpRequest"}
SLEEP = 1.2                                     # サイト負荷を抑える（1リクエストごとに待つ）
OUTDIR = Path(__file__).resolve().parent / "original"
COLS = ["開催日", "レース", "枠", "登番", "体重", "調整重量", "チルト", "展示タイム", "一周", "まわり足",
        "天候", "風向", "風速", "波高", "取得時刻"]


def num(s):
    s = (s or "").strip().replace("cm", "").replace("m", "")
    return float(s) if re.fullmatch(r"-?\d+(\.\d+)?", s) else None


def fetch_race(day, r, sess):
    """1レース分。展示前・非開催は []。日付・レースが要求と違うページなら [] を返す（取り違え防止）。"""
    res = sess.get(URL.format(d=day, r=r), headers=HEAD, timeout=20)
    res.raise_for_status(); res.encoding = "utf-8"; html = res.text
    time.sleep(SLEEP)
    if len(html) < 2000 or f"selected' data-day='{day}' data-race='{r}'" not in html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    tb = soup.find("table", class_="tbl_oriten")
    if tb is None:
        return []
    tobans = re.findall(r"[AB][12]\s*(\d{4})", soup.get_text(" "))[:6]
    sui = soup.find("table", class_="tbl_suimen")
    sv = [td.get_text("", strip=True) for td in sui.select("tbody td")] if sui else []
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    rows, trs = [], tb.select("tbody tr")
    for i in range(0, len(trs) - 1, 2):
        a = {td["class"][0]: td.get_text("", strip=True) for td in trs[i].find_all("td")}
        b = trs[i + 1].find("td")
        w = int(a.get("col1", "0") or 0)
        if not 1 <= w <= 6:
            continue
        rows.append({"開催日": day, "レース": r, "枠": w, "登番": tobans[w - 1] if len(tobans) == 6 else "",
                     "体重": num(a.get("col2")), "調整重量": num(b.get_text("", strip=True) if b else ""),
                     "チルト": num(a.get("col3")), "展示タイム": num(a.get("col4")),
                     "一周": num(a.get("col5")), "まわり足": num(a.get("col6")),
                     "天候": sv[0] if len(sv) > 0 else "", "風向": sv[1] if len(sv) > 1 else "",
                     "風速": num(sv[2]) if len(sv) > 2 else None, "波高": num(sv[3]) if len(sv) > 3 else None,
                     "取得時刻": now})
    return rows


def fetch_day(day, sess, overwrite=True):
    out = OUTDIR / f"orig_{day}.csv"
    if out.exists() and not overwrite:
        return "skip"
    first = fetch_race(day, 1, sess)
    if not first:
        return "none"                                # 非開催日（または1Rの展示前）
    rows = first + [x for r in range(2, 13) for x in fetch_race(day, r, sess)]
    OUTDIR.mkdir(exist_ok=True)
    pd.DataFrame(rows, columns=COLS).to_csv(out, index=False, encoding="utf-8-sig")
    return f"{len(rows) // 6}R"


def main(a):
    d0 = datetime.datetime.strptime(a[0], "%Y%m%d")
    d1 = datetime.datetime.strptime(a[1] if len(a) > 1 else a[0], "%Y%m%d")
    single = len(a) == 1
    sess = requests.Session()
    d = d0
    while d <= d1:
        day = d.strftime("%Y%m%d")
        try:
            st = fetch_day(day, sess, overwrite=single)
        except Exception as e:
            st = f"ERROR {e}"
        if st != "none" or single:
            print(day, st, flush=True)
        d += datetime.timedelta(days=1)


if __name__ == "__main__":
    main(sys.argv[1:])
