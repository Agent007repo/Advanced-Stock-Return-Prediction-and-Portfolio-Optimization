import os
import sys
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from stock_prediction_enhanced import StockPredictionModel

# Try to import tabulate for better formatting
try:
    from tabulate import tabulate
    has_tabulate = True
except ImportError:
    has_tabulate = False

def calculate_metrics(portfolio_results_file):
    """
    Calculate portfolio performance metrics from portfolio results CSV file
    
    Parameters:
    -----------
    portfolio_results_file : str
        Path to portfolio results CSV file
        
    Returns:
    --------
    dict of calculated metrics
    """
    # Check if file exists
    if not os.path.exists(portfolio_results_file):
        print(f"Error: File {portfolio_results_file} not found")
        return {}
    
    # Read portfolio results file
    try:
        data = pd.read_csv(portfolio_results_file)
        print(f"Loaded portfolio data with {len(data)} months of returns")
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return {}
    
    # Initialize metrics dictionary
    metrics = {}
    
    # Get strategy type from filename
    strategy_name = os.path.basename(portfolio_results_file).split('_')[-1].split('.')[0].capitalize()
    metrics["Strategy Type"] = strategy_name
    
    # Calculate average annualized return
    if 'strategy' in data.columns:
        avg_monthly_return = data['strategy'].mean()
        metrics["Average Annualized Return"] = (1 + avg_monthly_return) ** 12 - 1
        
        # Calculate annualized standard deviation
        metrics["Annualized Standard Deviation"] = data['strategy'].std() * np.sqrt(12)
        
        # Calculate Sharpe ratio
        if 'rf' in data.columns:
            # Use risk-free rate if available
            excess_returns = data['strategy'] - data['rf']
            metrics["Sharpe Ratio"] = excess_returns.mean() / excess_returns.std() * np.sqrt(12)
        else:
            # Use zero as risk-free rate if not available
            metrics["Sharpe Ratio"] = avg_monthly_return / data['strategy'].std() * np.sqrt(12)
        
        # Information ratio if market data is available
        if 'mkt_rf' in data.columns:
            excess_returns_market = data['strategy'] - data['mkt_rf']
            metrics["Information Ratio"] = excess_returns_market.mean() / excess_returns_market.std() * np.sqrt(12)
        else:
            metrics["Information Ratio"] = float('nan')
        
        # Calculate alpha from regression
        if 'mkt_rf' in data.columns:
            try:
                # Run CAPM regression with Newey-West standard errors
                capm_model = sm.ols(formula="strategy ~ mkt_rf", data=data).fit(
                    cov_type="HAC", cov_kwds={"maxlags": 3}, use_t=True
                )
                
                # Get alpha and annualize it (monthly to annual)
                monthly_alpha = capm_model.params["Intercept"]
                metrics["Annualized Alpha"] = (1 + monthly_alpha) ** 12 - 1
                metrics["Alpha t-statistic"] = capm_model.tvalues["Intercept"]
            except Exception as e:
                print(f"Warning: Error in alpha calculation: {str(e)}")
                metrics["Annualized Alpha"] = float('nan')
                metrics["Alpha t-statistic"] = float('nan')
        else:
            metrics["Annualized Alpha"] = float('nan')
            metrics["Alpha t-statistic"] = float('nan')
        
        # Calculate maximum one-month loss
        metrics["Maximum 1-Month Loss"] = data['strategy'].min()
        
        # Calculate drawdown
        data["cum_return"] = (1 + data['strategy']).cumprod()
        data["running_max"] = data["cum_return"].cummax()
        data["drawdown"] = 1 - data["cum_return"] / data["running_max"]
        metrics["Maximum Drawdown"] = data["drawdown"].max()
        
        # Calculate portfolio turnover if available (use average from model output)
        # This is a placeholder as turnover is calculated in the model directly
        metrics["Portfolio Turnover"] = float('nan')
    else:
        print("Error: No 'strategy' column found in data")
    
    return metrics

def display_metrics_table(metrics):
    """
    Display portfolio metrics in a formatted table
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of portfolio metrics
    """
    if not metrics:
        print("No metrics available")
        return
    
    print("\n========== PORTFOLIO PERFORMANCE METRICS ==========\n")
    
    # Format metrics as percentages where appropriate
    formatted_metrics = [
        ["Metric", "Value"],
        ["Strategy Type", metrics.get("Strategy Type", "N/A")],
        ["Average Annualized Return", f"{metrics.get('Average Annualized Return', float('nan')):.4%}"],
        ["Annualized Standard Deviation", f"{metrics.get('Annualized Standard Deviation', float('nan')):.4%}"],
        ["Sharpe Ratio", f"{metrics.get('Sharpe Ratio', float('nan')):.4f}"],
        ["Information Ratio", f"{metrics.get('Information Ratio', float('nan')):.4f}" if not np.isnan(metrics.get('Information Ratio', float('nan'))) else "N/A"],
        ["Annualized Alpha", f"{metrics.get('Annualized Alpha', float('nan')):.4%}" if not np.isnan(metrics.get('Annualized Alpha', float('nan'))) else "N/A"],
        ["Alpha t-statistic", f"{metrics.get('Alpha t-statistic', float('nan')):.4f}" if not np.isnan(metrics.get('Alpha t-statistic', float('nan'))) else "N/A"],
        ["Maximum Drawdown", f"{metrics.get('Maximum Drawdown', float('nan')):.4%}"],
        ["Maximum 1-Month Loss", f"{metrics.get('Maximum 1-Month Loss', float('nan')):.4%}"],
        ["Portfolio Turnover", f"{metrics.get('Portfolio Turnover', float('nan')):.4%}" if not np.isnan(metrics.get('Portfolio Turnover', float('nan'))) else "N/A"]
    ]
    
    # Display metrics as a nicely formatted table
    if has_tabulate:
        print(tabulate(formatted_metrics, headers="firstrow", tablefmt="grid"))
    else:
        # Basic formatting if tabulate is not available
        for metric, value in formatted_metrics:
            print(f"{metric:30} {value}")

def main():
    """
    Run the stock prediction model with different investment strategies
    and calculate comprehensive performance metrics
    """
    try:
        # Get work directory
        work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print(f"Working directory: {work_dir}")
        
        # Check for fast mode command line argument
        fast_mode = len(sys.argv) > 1 and sys.argv[1] == 'fast'
        
        # Define strategies to test
        strategies = [
            {"name": "mixed", "n_long": 70, "n_short": 30, "description": "Mixed strategy with 70 long, 30 short"},
            {"name": "long-short", "n_long": 50, "n_short": 50, "description": "Equal long-short with 50 stocks each"},
            {"name": "long-only", "n_long": 50, "n_short": 0, "description": "Long-only with top 50 stocks"}
        ]
        
        for strategy in strategies:
            print(f"\n{'='*80}")
            print(f"Running {strategy['description']}")
            print(f"{'='*80}")
            
            # Initialize and run model with this strategy
            model = StockPredictionModel(
                work_dir=work_dir,
                n_long=strategy["n_long"],
                n_short=strategy["n_short"],
                strategy_type=strategy["name"],
                fast_mode=fast_mode
            )
            
            # Run pipeline with Ridge model
            try:
                model.run_full_pipeline(model_for_portfolio="ridge")
                print(f"Completed {strategy['name']} strategy model run")
            except Exception as e:
                print(f"Error running model: {str(e)}")
                continue
            
            # Calculate and display comprehensive metrics from saved results
            # Try both potential file patterns (with and without 'enhanced')
            results_file = os.path.join(work_dir, f"portfolio_results_ridge_enhanced_{strategy['name']}.csv")
            if not os.path.exists(results_file):
                results_file = os.path.join(work_dir, f"portfolio_results_ridge_{strategy['name']}.csv")
                
            if os.path.exists(results_file):
                print(f"\nCalculating comprehensive metrics for {strategy['name']} strategy...")
                metrics = calculate_metrics(results_file)
                display_metrics_table(metrics)
            else:
                print(f"Warning: Results file not found for {strategy['name']} strategy. Checked both with and without 'enhanced' in filename.")
    
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 