"""Question 3 - when should the signal be switched off?

A rule-based kill switch on the monitoring metrics (config.GATE_RULES). For each
rebalance date t the gate reads only information available at t:

  * cross-sectional metrics (psi_score, auc_1w, ...) are computed in monitoring.py
    on the trailing window that ends the week *before* t - already point-in-time;
  * portfolio metrics (rolling_sharpe, drawdown) at row t include the return
    earned from t to t+1, so they are lagged one week here.

Champion  = the model's ungated weekly returns (as in portfolio_returns).
Challenger = same returns, but 0 (cash) in weeks where the gate is off, paying the
             one-way cost of liquidating / re-entering the whole book on each flip.
The comparison is made out-of-time from config.OOT_START.
"""
import numpy as np
import pandas as pd
import config
from sv import db

PORTFOLIO_METRICS = ("rolling_sharpe", "drawdown", "active_drawdown")
OPS = {">": np.greater, "<": np.less}


def metrics_wide(con, model: str) -> pd.DataFrame:
    m = db.read(con, "SELECT date, metric, value FROM monitoring WHERE model = $m", {"m": model})
    w = m.pivot(index="date", columns="metric", values="value").sort_index()
    lag = [c for c in PORTFOLIO_METRICS if c in w.columns]
    w[lag] = w[lag].shift(1)
    return w


def decide(w: pd.DataFrame, rules: dict = config.GATE_RULES) -> pd.DataFrame:
    """One row per date: gate_on and the comma-separated list of tripped rules."""
    tripped = pd.DataFrame(index=w.index)
    for metric, (op, thr) in rules.items():
        if metric in w.columns:
            tripped[metric] = OPS[op](w[metric], thr) & w[metric].notna()
    reason = tripped.apply(lambda r: ",".join(c for c in tripped.columns if r[c]), axis=1)
    return pd.DataFrame({"date": w.index, "gate_on": ~tripped.any(axis=1).values,
                         "reason": reason.replace("", None).values})


def challenger_returns(con, model: str, gate: pd.DataFrame, cost_bps: float = config.COST_BPS) -> pd.DataFrame:
    pr = db.read(con, "SELECT date, ret_gross, ret_net, turnover, n_held FROM portfolio_returns "
                      "WHERE strategy = $s ORDER BY date", {"s": model})
    g = gate.set_index("date")["gate_on"].reindex(pr.date).fillna(True).values   # no metrics yet -> stay on
    flip = np.abs(np.diff(g.astype(int), prepend=int(g[0])))                      # 1 on entry / exit
    out = pr.copy()
    out["ret_gross"] = np.where(g, pr.ret_gross, 0.0)
    out["ret_net"] = np.where(g, pr.ret_net, 0.0) - flip * cost_bps / 1e4
    out["turnover"] = np.where(g, pr.turnover, 0.0) + flip
    out["n_held"] = np.where(g, pr.n_held, 0)
    out["strategy"] = f"{model}_gated"
    return out


def save(con, model: str, gate: pd.DataFrame, chal: pd.DataFrame) -> None:
    con.execute("DELETE FROM gate_decisions WHERE model = ?", [model])
    db.write_df(con, gate.assign(model=model)[["date", "model", "gate_on", "reason"]], "gate_decisions")
    con.execute("DELETE FROM portfolio_returns WHERE strategy = ?", [f"{model}_gated"])
    db.write_df(con, chal[["date", "strategy", "ret_gross", "ret_net", "turnover", "n_held"]], "portfolio_returns")


def run(con, models: list[str]) -> pd.DataFrame:
    for m in models:
        gate = decide(metrics_wide(con, m))
        save(con, m, gate, challenger_returns(con, m, gate))
        off = gate[~gate.gate_on]
        print(f"{m}: gate off {len(off)}/{len(gate)} weeks; top reasons "
              f"{off.reason.value_counts().head(3).to_dict()}", flush=True)
    return db.run_sql_file(con, "champion_challenger", {"start": config.OOT_START})


if __name__ == "__main__":
    import sys
    con = db.connect()
    pd.set_option("display.width", 160)
    print(run(con, sys.argv[1:] or ["momentum", "gbm", "gbm_expected"]).round(3))
