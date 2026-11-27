"""展示タイムの取得（BoatraceOpenAPI previews ミラー。元データは公式の直前情報）。

公式 boatrace.jp の beforeinfo はこのPCから 403(Access Denied) になるため、
結果取得(buildResults.py)と同じ BoatraceOpenAPI の previews 版を使う。
  当日: https://boatraceopenapi.github.io/previews/v2/today.json
  過去: https://raw.githubusercontent.com/BoatraceOpenAPI/previews/HEAD/docs/v2/{YYYY}/{YYYYMMDD}.json
展示が未公開のレースはミラーに出ないか値が空 → 空欄のまま（補完しない）。
ミラーの更新間隔は未確認。締切直前に取り直すこと。
"""
import datetime
import json
import urllib.request

import numpy as np
import pandas as pd

URL_TODAY = "https://boatraceopenapi.github.io/previews/v2/today.json"
URL_DAY = "https://raw.githubusercontent.com/BoatraceOpenAPI/previews/HEAD/docs/v2/{y}/{d}.json"
COLS = ["場コード", "開催日", "レース", "枠", "展示タイム", "展示ST", "風速", "風向", "波高", "取得時刻"]


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "boatrace-model/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch(jcd, day):
    """jcd='13', day='20261127' → 展示タイムの DataFrame（1行=1艇）。"""
    jst = datetime.timezone(datetime.timedelta(hours=9))
    now = datetime.datetime.now(jst)
    url = URL_TODAY if day == now.strftime("%Y%m%d") else URL_DAY.format(y=day[:4], d=day)
    data = _get(url)["previews"]
    want = f"{day[:4]}-{day[4:6]}-{day[6:]}"
    rows = []
    for race in data:
        if race.get("race_date") != want or int(race.get("race_stadium_number", 0)) != int(jcd):
            continue   # 日付・場が違うものは使わない（取り違え防止）
        for b in race.get("boats", []):
            t = b.get("racer_exhibition_time")
            rows.append({"場コード": f"{int(jcd):02d}", "開催日": day, "レース": int(race["race_number"]),
                         "枠": int(b["racer_boat_number"]),
                         "展示タイム": t if t not in (None, 0, 0.0) else None,
                         "展示ST": b.get("racer_start_timing"),
                         "風速": race.get("race_wind"), "風向": race.get("race_wind_direction_number"),
                         "波高": race.get("race_wave"), "取得時刻": now.strftime("%Y-%m-%d %H:%M")})
    return pd.DataFrame(rows, columns=COLS).sort_values(["レース", "枠"]).reset_index(drop=True)


def to_array(df, rnos):
    """展示DataFrame → (レース数, 6) の配列。無い所は NaN（＝その艇は展示なし扱い）。"""
    a = np.full((len(rnos), 6), np.nan)
    if df is None or df.empty:
        return a
    for i, r in enumerate(rnos):
        for _, x in df[df.レース.astype(int) == int(r)].iterrows():
            v = pd.to_numeric(x.展示タイム, errors="coerce")
            if 1 <= int(x.枠) <= 6 and pd.notna(v):
                a[i, int(x.枠) - 1] = float(v)
    return a
