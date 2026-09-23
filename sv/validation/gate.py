"""Rule-based kill switch over the monitoring metrics (config.GATE_RULES).

A decision for signal date t uses only information available at t. Cross-sectional
metrics already end the week before t, while portfolio metrics carry the return earned
after t and are lagged two signal weeks here.

The champion is the ungated strategy; the challenger holds cash in weeks the gate is off,
with costs recomputed from the holdings actually traded. Both are compared out-of-time.
"""
import numpy as np
import pandas as pd
import config
from sv import db, backtest

PORTFOLIO_METRICS = ("rolling_sharpe", "drawdown", "active_drawdown")
OPS = {">": np.greater, "<": np.less}


def metrics_wide(con, model: str) -> pd.DataFrame:
    m = db.read(con, "SELECT date, metric, value FROM monitoring WHERE model = $m", {"m": model})
    w = m.pivot(index="date", columns="metric", values="value").sort_index()
    lag = [c for c in PORTFOLIO_METRICS if c in w.columns]
    w[lag] = w[lag].shift(config.MONITOR_RETURN_LAG)
    return w


def decide(w: pd.DataFrame, rules: dict = config.GATE_RULES) -> pd.DataFrame:
    """One row per date: gate_on and the comma-separated list of tripped rules."""
    tripped = pd.DataFrame(index=w.index)
    for metric, (op, thr) in rules.items():
        if metric in w.columns:
            tripped[metric] = OPS[op](w[metric], thr) & w[metric].notna()
    # A missing metric switches the strategy off.
    for metric in rules:
        tripped[f"missing:{metric}"] = w[metric].isna() if metric in w else True
    reason = tripped.apply(lambda r: ",".join(c for c in tripped.columns if r[c]), axis=1)
    return pd.DataFrame({"date": w.index, "gate_on": ~tripped.any(axis=1).values,
                         "reason": reason.replace("", None).values})


def challenger_returns(con, model: str, gate: pd.DataFrame, cost_bps: float = config.COST_BPS) -> pd.DataFrame:
    weights = backtest.top_n_weights(backtest._panel(con, model))
    on = gate.set_index("date")["gate_on"]
    # Costs come from the positions actually held while the gate is on.
    weights["w"] *= weights.date.map(on).fillna(False).astype(float)
    out = backtest.portfolio_returns(weights, cost_bps=cost_bps)
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
