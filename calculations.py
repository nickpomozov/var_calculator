# This script performs the core calculations for backtesting Value at Risk (VaR) and Expected Shortfall (ES) using historical stock data. It extracts data for a specified ticker, calculates log returns, and iteratively computes parametric and empirical VaR along with ES for a rolling window. The results are saved to a CSV file and a graph is generated for visualization. This script can be run independently to perform the calculations without the Streamlit interface.
# Financial Risk Management Project - Master SEP 2025-26


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def run_project():
    # 1.Data extraction
    ticker = "^FCHI"  
    print(f"Fetching data for {ticker}...")
    df = yf.download(ticker, period="5y")
    
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Calculate Log Returns
    returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()

    # 2. Parameters
    window = 504        
    test_period = 252   
    alpha_var = 0.99    
    alpha_es = 0.975    

    results = []

    # 3. Rolling calculations loop
    print("Running rolling calculations...")
    for i in range(len(returns) - test_period, len(returns)):
        train = returns.iloc[i-window:i]
        actual_return = returns.iloc[i]
        
        # Parametric VaR (Normal) 
        mu, sigma = train.mean(), train.std()
        var_param = -(mu + sigma * norm.ppf(1 - alpha_var))
        
        # Empirical VaR (Historical) 
        var_emp = -np.percentile(train, (1 - alpha_var) * 100)
        
        # Expected Shortfall (97.5% Historical)
        es_threshold = np.percentile(train, (1 - alpha_es) * 100)
        es_975 = -train[train <= es_threshold].mean()

        results.append({
            'Date': returns.index[i],
            'Return': actual_return,
            'Loss': -actual_return,
            'VaR_Param': var_param,
            'VaR_Emp': var_emp,
            'ES_975': es_975
        })

    res_df = pd.DataFrame(results).set_index('Date')

    # 4. Backtest logic
    res_df['Breach_Param'] = res_df['Loss'] > res_df['VaR_Param']
    res_df['Breach_Emp'] = res_df['Loss'] > res_df['VaR_Emp']

    # 5. Results summary
    total_days = len(res_df)
    print(f"\n--- BACKTESTING RESULTS ({total_days} days) ---")
    print(f"Parametric Breaches: {res_df['Breach_Param'].sum()} (Expected: {total_days * 0.01:.1f})")
    print(f"Empirical Breaches: {res_df['Breach_Emp'].sum()} (Expected: {total_days * 0.01:.1f})")
    
    # ES Backtest: Average Loss during breaches vs Predicted ES
    avg_breach_loss = res_df[res_df['Breach_Emp']]['Loss'].mean()
    avg_predicted_es = res_df['ES_975'].mean()
    print(f"Avg Loss on Breach: {avg_breach_loss:.4f}")
    print(f"Avg Predicted ES: {avg_predicted_es:.4f}")

    # 6. Visualisation
    plt.figure(figsize=(12, 6))
    plt.plot(res_df['Loss'], label='Actual Daily Loss', color='lightgray', alpha=0.7)
    plt.plot(res_df['VaR_Param'], label='VaR Parametric (99%)', color='blue', linestyle='--')
    plt.plot(res_df['VaR_Emp'], label='VaR Empirical (99%)', color='red')
    plt.title(f'VaR Backtesting: {ticker}')
    plt.legend()
    plt.savefig('VaR_Graph.png')
    print("\nGraph saved as 'VaR_Graph.png'")

    # Save data for submission
    res_df.to_csv("Financial_Risk_Project_Results.csv")
    print("Full results saved to 'Financial_Risk_Project_Results.csv'")

if __name__ == "__main__":
    run_project()