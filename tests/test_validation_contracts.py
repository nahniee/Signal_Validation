import unittest
import numpy as np
import pandas as pd
from sv import backtest
from sv.validation import monitoring, gate, overfit


class ValidationContracts(unittest.TestCase):
    def test_return_direction_threshold(self):
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler((-1, 1)).fit(np.array([[-.1], [.3]]))
        true = np.array([[-.01], [.01]])
        predicted = np.array([[.01], [.01]])
        a, b = scaler.transform(true), scaler.transform(predicted)
        self.assertEqual(np.mean(np.sign(a) == np.sign(b)), 1)
        self.assertEqual(np.mean(np.sign(a-scaler.min_) == np.sign(b-scaler.min_)), .5)

    def test_weight_drift_turnover(self):
        d = pd.to_datetime(['2022-01-07']*2 + ['2022-01-14']*2)
        w = pd.DataFrame({'date':d,'ticker':['A','B']*2,'w':[.5]*4,
                          'ret_fwd_1w_lag1':[1.,0.,0.,0.]})
        out = backtest.portfolio_returns(w, cost_bps=10)
        self.assertAlmostEqual(out.turnover.iloc[1], 1/6)
        self.assertAlmostEqual(out.ret_net.iloc[1], -1/3*.001)

    def test_missing_held_return_fails(self):
        w = pd.DataFrame({'date':[pd.Timestamp('2022-01-07')], 'ticker':['A'],
                          'w':[1.], 'ret_fwd_1w_lag1':[np.nan]})
        with self.assertRaises(ValueError):
            backtest.portfolio_returns(w)

    def test_psi_includes_tails(self):
        reference = np.linspace(-1,1,1000)
        shifted = reference + 100
        self.assertGreater(monitoring.psi(reference,shifted), 5)
        self.assertAlmostEqual(monitoring.psi(reference,reference),0)

    def test_missing_metrics_switch_off(self):
        dates = pd.to_datetime(['2022-01-07'])
        result = gate.decide(pd.DataFrame({'auc_1w':[.6]},index=dates))
        self.assertFalse(result.gate_on.iloc[0])
        self.assertIn('missing:psi_score',result.reason.iloc[0])

    def test_gate_reentry_charges_only_actual_trades(self):
        from unittest.mock import patch
        dates = pd.to_datetime(['2022-01-07','2022-01-14','2022-01-21'])
        panel = pd.DataFrame({'date':dates,'ticker':['A']*3,'score':[1.]*3,
                              'ret_fwd_1w':[0.]*3,'ret_fwd_1w_lag1':[0.]*3})
        decisions = pd.DataFrame({'date':dates,'gate_on':[True,False,True]})
        with patch('sv.backtest._panel',return_value=panel):
            result = gate.challenger_returns(None,'test',decisions,cost_bps=10)
        np.testing.assert_allclose(result.ret_net,[-.001,-.001,-.001])

    def test_exact_execution_calendar(self):
        import duckdb
        from pathlib import Path
        c=duckdb.connect(':memory:')
        c.execute(Path('sql/schema.sql').read_text())
        dates=pd.bdate_range('2021-12-01','2022-02-01')
        rows=[]
        for ticker in ['SPY','A']:
            for i,d in enumerate(dates):
                if ticker=='A' and d == pd.Timestamp('2022-01-17'):
                    continue
                rows.append([d,ticker,100+i,100+i,100+i,100+i,100+i,1e6])
        c.register('px',pd.DataFrame(rows,columns=['date','ticker','open','high','low','close','adj_close','volume']))
        c.execute('insert into prices select * from px')
        c.execute(Path('sql/queries/build_panel.sql').read_text(),{'start':'2022-01-07','end':'2022-02-01','min_price':5,'min_adv':5e6})
        row=c.execute("select ret_fwd_1w_lag1 from panel where ticker='A' and date='2022-01-07'").fetchone()
        self.assertIsNone(row[0]) # no substitution of Tuesday for the missing Monday exit
        actual=c.execute("select ret_fwd_1w_lag1 from panel where ticker='SPY' and date='2022-01-07'").fetchone()[0]
        prices={d:100+i for i,d in enumerate(dates)}
        self.assertAlmostEqual(actual,prices[pd.Timestamp('2022-01-17')]/prices[pd.Timestamp('2022-01-10')]-1)

    def test_cutoff_blocks_all_future_prices(self):
        import duckdb
        from pathlib import Path
        from unittest.mock import patch
        from sv import features
        c=duckdb.connect(':memory:')
        c.execute(Path('sql/schema.sql').read_text())
        dates=pd.bdate_range('2026-07-01','2026-09-15')
        frame=pd.DataFrame({'date':dates,'ticker':'SPY','open':100.,'high':100.,
                            'low':100.,'close':100.,'adj_close':100.,'volume':1e6})
        c.register('px',frame); c.execute('INSERT INTO prices SELECT * FROM px')
        with patch('config.EVALUATION_END','2026-08-28'):
            features.build_panel(c)
            before=c.execute('SELECT * FROM panel ORDER BY date').df()
            # Every price after the cutoff can change without affecting a single panel value.
            c.execute("UPDATE prices SET close=10000, adj_close=10000 WHERE date>'2026-08-28'")
            features.build_panel(c)
            pd.testing.assert_frame_equal(before,c.execute('SELECT * FROM panel ORDER BY date').df())
            self.assertEqual(features.load_wide(c,'adj_close').index.max(),pd.Timestamp('2026-08-28'))
        self.assertEqual(pd.Timestamp(c.execute("SELECT MAX(date) FROM panel WHERE ret_fwd_1w_lag1 IS NOT NULL").fetchone()[0]),pd.Timestamp('2026-08-14'))
        for column in ['ret_fwd_1w','ret_fwd_1w_lag1','ret_fwd_13w']:
            self.assertIsNone(c.execute(f"SELECT {column} FROM panel WHERE date='2026-08-28'").fetchone()[0])

    def test_oot_filter(self):
        import duckdb
        c=duckdb.connect(':memory:')
        c.execute('create table portfolio_returns(date DATE,strategy VARCHAR,ret_net DOUBLE)')
        c.execute("insert into portfolio_returns values ('2021-12-31','m',1),('2021-12-31','universe_ew',0),('2022-01-07','m',.1),('2022-01-07','universe_ew',.02)")
        a=overfit.active_returns(c,'m')
        self.assertEqual(len(a),1)
        self.assertAlmostEqual(a.iloc[0],.08)

if __name__ == '__main__':
    unittest.main()
