# Methodology and Limitations

## Project Type

Quantitative ML research project with out-of-sample portfolio evaluation.

## What This Project Demonstrates

- Cross-sectional return prediction using stock characteristics.
- Comparison of regularized linear models and neural-network approaches.
- Portfolio construction across long-only, long-short, and mixed strategies.
- Out-of-sample performance reporting for 2020-2023.
- Risk metrics such as Sharpe ratio, alpha, drawdown, and turnover.

## Important Limitations

This project should be read as an analytical and educational portfolio project, not as investment advice or a live trading system.

Potential limitations to validate before any production or investment use:

- Data availability and point-in-time correctness.
- Survivorship bias and delisting effects.
- Transaction costs, borrow costs, slippage, and market impact.
- Liquidity constraints and real-world shorting limitations.
- Regime sensitivity during unusually volatile market periods.
- Robustness of alpha after additional validation windows.

## Recommended Next Improvements

- Add a small quick-run mode with a reduced sample dataset.
- Add explicit data source documentation.
- Add transaction-cost sensitivity analysis.
- Add model card with assumptions and evaluation scope.
- Add a single command that regenerates the main result tables and plots.
