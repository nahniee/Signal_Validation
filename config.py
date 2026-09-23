"""Modelling and evaluation assumptions in one place, cited by the validation report."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "signal_validation.duckdb"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

# Sibling checkout holding the CLAM training code and weights under review.
QMR_DIR = ROOT.parent / "Quant_Model_Research"

# Universe: current top-N US listings by market cap, so delisted names are absent.
UNIVERSE_SIZE = 3000
UNIVERSE_ASOF = "2026-09-15"
EXCHANGES = ["NMS", "NYQ", "NGM", "NCM", "ASE"]

# Price history, with two warm-up years before the first signal.
PRICE_START = "2013-01-01"
PRICE_END = "2026-08-29"        # exclusive in yfinance: last observation is 2026-08-28

# Backtest: signal at week-end close, traded at the next session close.
BACKTEST_START = "2015-01-02"
HOLD_TOP_N = 50                 # long-only, equal weight
COST_BPS = 10                   # one-way commission, spread and slippage
BENCHMARK = "SPY"

# Tradability filter applied at every signal date.
MIN_PRICE = 5.0
MIN_ADV_USD = 5_000_000         # 20-day average dollar volume

# Candidate parameters, fixed rather than tuned here.
MOM_LOOKBACK, MOM_SKIP = 252, 21          # 12-1 momentum
GBM_LOOKBACK, GBM_HORIZON = 730, 63       # calendar-day lookback, original 63-step path
CLAM_SEQ_LEN = 252
CLAM_ORIGINAL_H5 = QMR_DIR / "quarterly_model.h5"
CLAM_ORIGINAL_SCALER = QMR_DIR / "quarterly_scaler.pkl"
CLAM_RETRAIN_CUTOFF = "2021-12-31"        # cutoff for the retrained twin

# Validation window and test settings.
OOT_START = "2022-01-03"
EVALUATION_END = "2026-08-28"   # holding periods unfinished by this date are excluded
N_TRIALS_DSR = 17               # 14 historical trials plus the 94/500/3000 universe-size runs
EXECUTION_RETURN = "ret_fwd_1w_lag1"
MONITOR_RETURN_LAG = 2          # the last delayed holding period is incomplete at signal close
BOOTSTRAP_BLOCK = 13            # weeks (~1 quarter)
BOOTSTRAP_N = 5000
PSI_BUCKETS = 10
MONITOR_WINDOW = 52             # rolling weeks for PSI, KS and AUC
VALIDATION_MODELS = ["momentum", "gbm", "gbm_expected", "gbm_weekly", "clam_2021",
                     "clam_weekly_cs_demeaned", "clam_weekly_cs_rank_small",
                     "clam_weekly_cs_rank_small_n94_seed20260922",
                     "clam_weekly_cs_rank_small_n500_seed20260922",
                     "clam_weekly_cs_rank_small_n3000_seed20260922"]

# Kill switch: the signal is held in cash for the coming week if any rule trips.
# Thresholds follow documented conventions; historical preregistration is not established.
GATE_RULES = {
    "psi_score":       (">", 0.25),   # score distribution shifted vs the development sample
    "rolling_sharpe":  ("<", 0.0),    # trailing 52w active Sharpe negative
    "auc_1w":          ("<", 0.50),   # no discriminatory power over the trailing 52w
    "active_drawdown": ("<", -0.15),  # relative drawdown beyond tolerance
}
