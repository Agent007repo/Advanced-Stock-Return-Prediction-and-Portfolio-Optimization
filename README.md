# Advanced Stock Return Prediction and Portfolio Optimization

## FINE 695 Individual Assignment

### Author: Samarth Basavaraj Annigeri

## Executive Summary

This project implements a comprehensive quantitative investment framework that employs machine learning algorithms for stock return prediction and portfolio construction. The implementation rigorously adheres to industry-standard methodologies while incorporating advanced risk management techniques and factor-based alpha enhancements. The evaluation focuses specifically on out-of-sample performance from January 2020 through December 2023, as required by the assignment specifications.

The mixed strategy (70% long, 30% short) achieved the best risk-adjusted returns with an annualized alpha of 7.71%, a Sharpe ratio of 0.71, and an annualized return of 11.71% during the evaluation period. These results demonstrate the effectiveness of the machine learning approach and portfolio construction methodology.

## Methodological Framework

### Data Processing and Feature Engineering

The model processes a comprehensive financial dataset containing 147 stock characteristics to predict future returns:

- **Cross-sectional Normalization**: Implements rank-based standardization to ensure comparability across stocks and mitigate the impact of outliers
- **Expanding Window Approach**: Employs a time-series cross-validation methodology to prevent look-ahead bias while maximizing training data utilization
- **Feature Importance Analysis**: Identifies and prioritizes the most significant predictive factors through rigorous statistical analysis

### Machine Learning Implementation

The framework implements multiple prediction models, each capturing different aspects of return predictability:

| Model | Description | Out-of-Sample R² |
|-------|-------------|------------------|
| Linear Regression (OLS) | Baseline model for comparative analysis | 0.00338 |
| LASSO Regression | L1 regularization for sparse feature selection | 0.00371 |
| Ridge Regression | L2 regularization for multicollinearity management | 0.00402 |
| Elastic Net | Combined L1/L2 regularization for balanced feature selection | 0.00375 |
| Neural Network | Dual-pathway architecture with residual connections | 0.00560-0.00910* |

*Neural Network R² varied by strategy, with the highest value observed in the long-short implementation.

### Advanced Neural Network Architecture

The implemented neural network features a dual-pathway architecture with residual connections:

- **Dual Information Pathways**: Parallel processing streams capture both linear and non-linear relationships
- **Residual Connections**: Enable efficient gradient flow during training and improve model convergence
- **Batch Normalization**: Stabilizes training and accelerates convergence
- **Dropout Regularization**: Prevents overfitting by randomly deactivating neurons during training

### Portfolio Construction Strategies

Three distinct portfolio construction methodologies were implemented and evaluated:

| Strategy | Description | Annualized Return | Sharpe Ratio | Alpha | Maximum Drawdown |
|----------|-------------|-------------------|--------------|-------|------------------|
| Mixed (70/30) | 70% allocation to long positions, 30% to short positions | 11.71% | 0.71 | 7.71% | 20.74% |
| Long-Short (50/50) | Equal allocation to 50 long and 50 short positions | 20.11% | 0.55 | 23.85% | 59.34% |
| Long-Only | Exclusive allocation to top 50 stocks with highest predicted returns | 15.76% | 0.58 | 2.90% | 25.20% |

## Quantitative Enhancements

### Risk Management Framework

- **Volatility-Based Position Sizing**: Dynamic capital allocation inversely proportional to historical stock volatility
- **Position Size Constraints**: Maximum individual position size capped at 3% to mitigate concentration risk
- **Volatility Targeting**: Portfolio construction methodology aims for consistent risk exposure across market regimes

### Factor-Based Alpha Enhancement

- **Factor Residualization**: Extracts idiosyncratic returns by controlling for known systematic risk factors
- **Style Neutralization**: Reduces unintended exposures to common factors including size, value, and momentum
- **Statistical Significance Enhancement**: Improves signal-to-noise ratio by isolating stock-specific alpha

## Comprehensive Performance Metrics

The framework produces a thorough set of performance metrics for rigorous evaluation:

### Risk-Adjusted Performance
- **Annualized Alpha**: Excess return after controlling for market exposure (CAPM)
- **Alpha t-statistic**: Statistical significance of the alpha estimate
- **Information Ratio**: Risk-adjusted excess return relative to benchmark
- **Sharpe Ratio**: Risk-adjusted return using standard deviation as risk measure

### Risk Metrics
- **Annualized Standard Deviation**: Volatility of monthly returns, annualized
- **Maximum Drawdown**: Worst peak-to-trough decline over the evaluation period
- **Maximum One-Month Loss**: Worst single-month performance

### Implementation Metrics
- **Portfolio Turnover**: Average monthly position change rate
- **Top Holdings Analysis**: Identification and analysis of most frequently selected positions

## Technical Implementation

### Code Architecture

The implementation follows object-oriented programming principles with modular design:

- **`StockPredictionModel`**: Core class implementing data processing, model training, and portfolio construction
- **`run_investment_strategy.py`**: Script for comparative analysis of multiple investment strategies
- **Supporting visualization modules**: Specialized code for neural network visualization and holdings analysis

### Required Dependencies

- **Core Libraries**: numpy, pandas, matplotlib, statsmodels
- **Machine Learning**: scikit-learn, tensorflow (for neural networks)
- **Visualization**: tabulate (for formatted output)

## Usage Instructions

### Standard Execution

```bash
python enhanced_code/run_investment_strategy.py
```

This executes the full analysis pipeline for all three portfolio strategies, producing comprehensive performance metrics and visualization outputs.

### Fast Mode Execution

```bash
python enhanced_code/run_investment_strategy.py fast
```

Utilizes an optimized parameter search space for more efficient execution during development and testing.

### Programmatic API

```python
from stock_prediction_enhanced import StockPredictionModel

# Initialize model with specific parameters
model = StockPredictionModel(
    n_long=70,             # Number of long positions
    n_short=30,            # Number of short positions
    strategy_type="mixed", # Strategy type: "mixed", "long-short", or "long-only"
    fast_mode=True         # Optional: Use reduced parameter search for faster execution
)

# Execute the full prediction and portfolio analysis pipeline
results = model.run_full_pipeline()
```

## Output Files

The framework generates the following output files:

1. **Prediction Data**:
   - `stock_predictions.csv`: Comprehensive predictions from all models

2. **Performance Analysis**:
   - `portfolio_results_ridge_enhanced_[strategy].csv`: Detailed performance metrics for each strategy
   - `performance_plot_ridge_enhanced_[strategy].png`: Visualization of cumulative performance vs. S&P 500

3. **Visualization**:
   - `neural_network_architecture.png`: Detailed visualization of the neural network architecture
   - `top_holdings_[strategy].png`: Graphical representation of the strategy's most significant positions


## Conclusion

This project demonstrates the successful application of machine learning techniques to stock return prediction and portfolio optimization. The mixed strategy achieved the best risk-adjusted performance with significant alpha generation and controlled drawdowns during the evaluation period. The implementation satisfies all assignment requirements while incorporating advanced quantitative techniques typically found in professional investment management.

The results highlight the effectiveness of combining multiple prediction models with sophisticated portfolio construction and risk management methodologies. The framework provides a solid foundation for further research and practical application in quantitative investment management.
