import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the predictions file
predictions = pd.read_csv('/Users/samarthbasavarajannigeri/Downloads/Prof Russ Individual Assignment/stock_predictions.csv')

# Focus on the first month of 2020 evaluation period
eval_period = predictions[(predictions['year'] == 2020) & (predictions['month'] == 1)].copy()

# Sort by ridge predictions (which is what we used for portfolio construction)
sorted_stocks = eval_period.sort_values('ridge', ascending=False)

# Get top 10 stocks for long positions
top_10_long = sorted_stocks.head(10)

# Create simplified position weights (equal weight with minor variation for visualization)
# In reality, our portfolio uses volatility-based position sizing but we're simplifying for the visualization
weights = np.linspace(12, 8, 10)  # Decreasing weights from 12% to 8%
weights = weights / sum(weights) * 100  # Normalize to percentages

# Assign weights to top stocks
top_10_long = top_10_long.reset_index(drop=True)
top_10_long['position_size'] = weights

# Add company ticker/name with PERMNO for identification in a more readable format
top_10_long['ticker'] = [f"Stock {i+1} (#{top_10_long.iloc[i]['permno']})" for i in range(10)]

# Create a bar chart of the top 10 holdings with improved formatting
plt.figure(figsize=(14, 7))  # Wider figure for better spacing
colors = plt.cm.Blues(np.linspace(0.6, 0.9, 10))

# Use width parameter to create space between bars
plt.bar(top_10_long['ticker'], top_10_long['position_size'], color=colors, width=0.7)

plt.title('Top 10 Holdings in Mixed Strategy Portfolio (Jan 2020)', fontsize=16)
plt.xlabel('Stock Identifier', fontsize=12)
plt.ylabel('Portfolio Weight (%)', fontsize=12)

# Improve x-axis label formatting
plt.xticks(fontsize=10, rotation=45, ha='right')
plt.subplots_adjust(bottom=0.2)  # Make room for rotated labels

plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add data labels on top of bars
for i, value in enumerate(top_10_long['position_size']):
    plt.text(i, value + 0.5, f'{value:.1f}%', ha='center', fontweight='bold')

# Add PERMNO and predicted return as a note at the bottom
plt.figtext(0.5, 0.01, 
            "Note: Based on top predicted returns from ridge regression model. Actual portfolio uses volatility-adjusted position sizing.", 
            ha='center', fontsize=9, style='italic')

plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to make room for note
plt.savefig('/Users/samarthbasavarajannigeri/Downloads/Prof Russ Individual Assignment/top_10_holdings.png', dpi=300)
print("Chart saved as 'top_10_holdings.png'")

# Print top 10 stocks details for reference
print("\nTop 10 stocks details:")
print(top_10_long[['permno', 'ridge', 'position_size']].to_string(index=False))
