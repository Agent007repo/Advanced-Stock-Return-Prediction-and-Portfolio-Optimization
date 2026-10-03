import unittest
import numpy as np
import pandas as pd
from portfolio_weights import capped_inverse_volatility, weighted_monthly_returns

class PortfolioRegressionTests(unittest.TestCase):
    def test_cap_survives_redistribution(self):
        w=capped_inverse_volatility([.001]+[.2]*49)
        self.assertLessEqual(w.max(),.03+1e-12)
        self.assertAlmostEqual(w.sum(),1.)
    def test_infeasible_cap_leaves_cash(self):
        w=capped_inverse_volatility([.1,.2],budget=1.)
        np.testing.assert_allclose(w,[.03,.03])
    def test_short_contribution_has_correct_sign(self):
        h=pd.DataFrame({'year':[2020,2020],'month':[1,1],'permno':[1,2],
             'portfolio':['long','short'],'weight':[.7,-.3],'stock_exret':[.1,.2]})
        self.assertAlmostEqual(weighted_monthly_returns(h).strategy.iloc[0],.01)
    def test_invalid_volatility_rejected(self):
        for value in [0.,-1.,np.nan]:
            with self.assertRaises(ValueError): capped_inverse_volatility([value])
    def test_overlap_rejected(self):
        h=pd.DataFrame({'year':[2020,2020],'month':[1,1],'permno':[1,1],
             'portfolio':['long','short'],'weight':[.7,-.3],'stock_exret':[.1,.2]})
        with self.assertRaises(ValueError): weighted_monthly_returns(h)

class PublicPipelineRegressionTests(unittest.TestCase):
    def test_failed_pipeline_propagates_error(self):
        import contextlib,io
        from stock_prediction_enhanced import StockPredictionModel
        model=object.__new__(StockPredictionModel)
        def failure(): raise FileNotFoundError('required licensed input')
        model.load_data=failure
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(FileNotFoundError): model.run_full_pipeline()
