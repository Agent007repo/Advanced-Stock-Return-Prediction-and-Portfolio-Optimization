"""Capped inverse-volatility allocation with cash when a cap is infeasible."""
import numpy as np
import pandas as pd


def capped_inverse_volatility(volatilities, budget=1.0, cap=.03):
    values = np.asarray(volatilities, dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Volatilities must be finite and strictly positive.')
    if not np.isfinite(budget) or not np.isfinite(cap) or budget < 0 or cap <= 0:
        raise ValueError('Invalid allocation budget or cap.')
    result = np.zeros(len(values))
    available = np.arange(len(values))
    remaining = min(budget, len(values) * cap)
    while len(available) and remaining > 1e-12:
        inverses = 1 / values[available]
        proposed = remaining * inverses / inverses.sum()
        saturated = proposed >= cap
        if not saturated.any():
            result[available] = proposed
            break
        result[available[saturated]] = cap
        remaining -= cap * saturated.sum()
        available = available[~saturated]
    return result


def weighted_monthly_returns(holdings, return_column='stock_exret'):
    if holdings.duplicated(['year', 'month', 'permno']).any():
        raise ValueError('Long and short holdings must be disjoint within each month.')
    h = holdings.copy()
    h['contribution'] = h['weight'] * h[return_column]
    returns = h.groupby(['year', 'month', 'portfolio'])['contribution'].sum().unstack(fill_value=0)
    returns['strategy'] = returns.sum(axis=1)
    return returns.reset_index()
