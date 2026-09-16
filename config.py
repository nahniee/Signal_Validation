"""Project-wide constants. Everything that is a modelling *assumption* lives here
so the validation report can cite a single source of truth."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "signal_validation.duckdb"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

# Sibling checkout of the original model repo (deployed CLAM weights + training code)
QMR_DIR = ROOT.parent / "Quant_Model_Research"

# --- Universe --------------------------------------------------------------
UNIVERSE_SIZE = 3000            # top-N US-listed equities by market cap (Russell 3000 proxy)
UNIVERSE_ASOF = "2026-09-15"    # screener snapshot date -> survivorship bias, see report §Limitations
EXCHANGES = ["NMS", "NYQ", "NGM", "NCM", "ASE"]

# --- Price history ---------------------------------------------------------
PRICE_START = "2013-01-01"      # 2y warm-up before backtest start
PRICE_END = "2026-09-15"      # exclusive in yfinance -> last bar 2026-09-14 (snapshot frozen for reproducibility)

# --- Backtest ----------------------------------------------------------------
BACKTEST_START = "2015-01-02"
REBALANCE = "W-FRI"             # weekly, signal computed on Friday close, traded next open
HOLD_TOP_N = 50                 # long-only, equal weight
COST_BPS = 10                   # one-way: 5 bps commission/spread + 5 bps slippage
BENCHMARK = "SPY"

# Liquidity filter applied at every rebalance date (tradability, not alpha)
MIN_PRICE = 5.0
MIN_ADV_USD = 5_000_000         # 20-day average dollar volume

# --- Candidate models (fixed, no tuning: this repo is the judge, not the modeller)
MOM_LOOKBACK, MOM_SKIP = 252, 21          # classic 12-1 momentum
GBM_LOOKBACK, GBM_HORIZON = 504, 65       # 2y daily lookback, 65-day path: Long_Term_Trading.get_gbm_path_simulation
CLAM_SEQ_LEN, CLAM_HORIZON = 252, 65      # matches Quant_Model_Research CONFIG['quarterly']
CLAM_ORIGINAL_H5 = QMR_DIR / "quarterly_model.h5"
CLAM_ORIGINAL_SCALER = QMR_DIR / "quarterly_scaler.pkl"
CLAM_ORIGINAL_TRAIN_END = "2025-08-22"    # file mtime; data before this is in-sample
CLAM_RETRAIN_CUTOFF = "2021-12-31"        # retrained twin -> ~4.7y out-of-time

# --- Validation -------------------------------------------------------------
OOT_START = "2022-01-03"        # out-of-time window for champion/challenger
N_TRIALS_DSR = 3                # number of candidate strategies compared (for DSR)
BOOTSTRAP_BLOCK = 13            # weeks (~1 quarter) for stationary block bootstrap
BOOTSTRAP_N = 5000
PSI_BUCKETS = 10
MONITOR_WINDOW = 52             # rolling weeks for PSI/KS/AUC

# --- Kill switch (gate) -------------------------------------------------------
# Signal is switched OFF for the coming week if ANY rule trips. Thresholds follow
# common model-monitoring conventions and were fixed before the OOT evaluation.
GATE_RULES = {
    "psi_score":       (">", 0.25),   # score distribution has shifted vs development sample
    "rolling_sharpe":  ("<", 0.0),    # trailing 52w active Sharpe negative
    "auc_1w":          ("<", 0.50),   # no discriminatory power over trailing 52w
    "active_drawdown": ("<", -0.15),  # relative drawdown vs universe beyond tolerance
}
