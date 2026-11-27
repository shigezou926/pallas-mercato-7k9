from pathlib import Path
JCD = "13"
VENUE = "尼崎"
import os
KDATA = Path(os.environ.get("KDATA") or Path(__file__).resolve().parents[3] / "kdata")   # 読み取り専用
NEW_FROM = "260719"                    # これ以降は追記分＝モデル作成後に手に入った未使用データ(新確認)
USE_ORIGINAL = os.environ.get("ORIGINAL") == "1"   # オリジナル展示(一周・まわり足)入りモデル
ORIG_DIR = Path(__file__).resolve().parent / "original"   # fetch_original.py の保存先
OUT = Path(__file__).resolve().parent / ("results_original" if USE_ORIGINAL else "results")
OUT.mkdir(exist_ok=True)
TRAIN_DAYS, TUNE_DAYS = 100, 30        # 残り(約33開催日)が確認用。ここは最後に1回だけ見る
WARMUP_DAYS = 15                       # 学習の最初の15日は選手履歴が空なので学習行から除く
N_SIM = 400
# 採用ルール(事前に固定): 確認期間で ①Brierが両ベースラインより良い ②購入30点以上で回収率100%以上
MIN_HOLDOUT_BETS = 30
