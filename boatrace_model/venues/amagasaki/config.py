from pathlib import Path
JCD = "13"
VENUE = "尼崎"
KDATA = Path(__file__).resolve().parents[3] / "kdata"      # 読み取り専用
OUT = Path(__file__).resolve().parent / "results"
TRAIN_DAYS, TUNE_DAYS = 100, 30        # 残り(約33開催日)が確認用。ここは最後に1回だけ見る
WARMUP_DAYS = 15                       # 学習の最初の15日は選手履歴が空なので学習行から除く
N_SIM = 400
# 採用ルール(事前に固定): 確認期間で ①Brierが両ベースラインより良い ②購入30点以上で回収率100%以上
MIN_HOLDOUT_BETS = 30
