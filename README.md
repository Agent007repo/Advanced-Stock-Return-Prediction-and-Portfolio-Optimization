# Stock Return Prediction and Portfolio Construction

FINE 695 project by Samarth Basavaraj Annigeri. Python scripts compare monthly stock-return predictors using expanding training windows and construct ranked long-only, long-short, and mixed portfolios. `StockPredictionModel` implements the enhanced workflow; legacy scripts retain simpler decile analysis.

## Data and execution

The required licensed source file `mma_sample_v2.csv` is not included. Supply it in the repository directory or pass a data directory as `work_dir`. It must contain `date`, `permno`, `year`, `month`, `stock_exret`, and the predictors listed in `factor_char_list.csv`. Optional `mkt_ind.csv` supplies monthly risk-free and market excess returns. Check units and availability dates before using the data.

Use Python 3.11+ and install `requirements.txt`. TensorFlow is optional and must be explicitly installed via `requirements-nn.txt`; the program never installs it automatically.

```bash
pip install -r requirements.txt
python run_investment_strategy.py fast
# Full linear models, plus the neural model if TensorFlow is available:
python run_investment_strategy.py
```

Fast mode runs Ridge only with a reduced parameter grid. The evaluation window is January 2020–December 2023. Empty training/validation/test windows are skipped. Failed strategies cause a nonzero exit status.

```python
from stock_prediction_enhanced import StockPredictionModel
model = StockPredictionModel(work_dir="/path/to/licensed/data", n_long=70,
                             n_short=30, strategy_type="mixed", fast_mode=True)
results = model.run_full_pipeline()
```

## Portfolio semantics

Cross-sectional feature ranks precede annual expanding-window prediction. Portfolio formation defaults to a predetermined Ridge baseline; test results do not select the portfolio model. Historical monthly volatility uses observations strictly before portfolio formation and annualizes with sqrt(12). Inverse-volatility weights are redistributed subject to a 3% absolute per-position cap. If too few positions exist to spend a budget under the cap, the remainder stays unallocated.

Long and short selections are disjoint. Long-only has a 100% long budget; mixed has 70% long and 30% short; long-short has 100% long and 100% short gross budgets. Short holdings carry negative weights. Portfolio excess returns are the sum of saved holding weights times `stock_exret`, rather than unweighted averages. Sharpe ratios do not subtract the risk-free rate twice. Drawdown includes initial wealth. This does not model financing, slippage, borrow constraints, or transaction costs.

## Outputs and status

The pipeline writes `stock_predictions.csv`, strategy/model portfolio results and performance plots, and `portfolio_holdings_<model>_<strategy>.csv`. `generate_top_holdings_v2.py` plots these actual holdings. It requires those files and never invents holdings.

Earlier alpha, Sharpe, return, model-R², and drawdown tables are withdrawn pending a licensed-data rerun. They cannot validate the corrected weights and metrics. Existing CSV/PNG artifacts are historical and may disagree with current code. Residualization and factor adjustments in the implementation are heuristics; they do not establish unbiased alpha, significance, or professional deployment readiness.

```bash
python -m unittest discover -s tests -p test_regressions.py -v
```

Six regression tests check the cap after redistribution, infeasible budgets, short-return sign, invalid volatility, overlapping holdings, and error propagation. The class imports without TensorFlow and reports missing source data clearly. Licensed-data prediction, neural training, and financial performance remain unverified.
