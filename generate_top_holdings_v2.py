"""Visualize the actual saved mixed-strategy long holdings."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parent
    path = root / 'portfolio_holdings_ridge_mixed.csv'
    if not path.is_file():
        raise FileNotFoundError('Run run_investment_strategy.py first to generate actual holdings.')
    holdings = pd.read_csv(path)
    selected = holdings.loc[(holdings['year'] == 2020) & (holdings['month'] == 1) &
                            (holdings['portfolio'] == 'long')].nlargest(10, 'weight')
    if selected.empty:
        raise ValueError('No long holdings for January 2020.')
    plt.figure(figsize=(12, 6))
    plt.bar(selected['permno'].astype(str), selected['weight'] * 100)
    plt.ylabel('Actual portfolio weight (%)')
    plt.xlabel('PERMNO')
    plt.title('Top 10 Mixed-Strategy Long Holdings (January 2020)')
    plt.tight_layout()
    plt.savefig(root / 'top_10_holdings.png', dpi=150)


if __name__ == '__main__':
    main()
