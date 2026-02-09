import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

st.set_page_config(page_title="Financial Risk Management Project", layout="wide")

st.title("📊 VaR & ES Backtesting Dashboard")
st.subheader("Master SEP 2025-26 - Université de Reims")

# 1. Sidebar Configuration
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker Symbol", value="^FCHI") # CAC 40
    window = st.slider("Rolling Window (Days)", 252, 756, 504) 
    test_period = 252 # Last year
    
# 2. Data Extraction
@st.cache_data
def load_data(symbol):
    df = yf.download(symbol, period="5y")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

data_load_state = st.text('Loading data from Yahoo Finance...')
raw_data = load_data(ticker)
data_load_state.text('Data loaded successfully!')

# 3. Calculations
returns = np.log(raw_data['Close'] / raw_data['Close'].shift(1)).dropna()

results = []
for i in range(len(returns) - test_period, len(returns)):
    train = returns.iloc[i-window:i]
    actual_loss = -returns.iloc[i]
    
    # Parametric Normal VaR (99%)
    mu, sigma = train.mean(), train.std()
    var_param = -(mu + sigma * norm.ppf(0.01))
    
    # Empirical VaR (99%)
    var_emp = -np.percentile(train, 1)
    
    # Expected Shortfall (97.5%)
    es_threshold = np.percentile(train, 2.5)
    es_975 = -train[train <= es_threshold].mean()

    results.append({
        'Date': returns.index[i],
        'Loss': actual_loss,
        'VaR_Param': var_param,
        'VaR_Emp': var_emp,
        'ES_975': es_975
    })

res_df = pd.DataFrame(results).set_index('Date')
res_df['Breach_Param'] = res_df['Loss'] > res_df['VaR_Param']
res_df['Breach_Emp'] = res_df['Loss'] > res_df['VaR_Emp']

# 4. Display Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Parametric Breaches", res_df['Breach_Param'].sum(), f"Expected: {test_period*0.01:.1f}")
with col2:
    st.metric("Empirical Breaches", res_df['Breach_Emp'].sum(), f"Expected: {test_period*0.01:.1f}")
with col3:
    st.metric("Avg Predicted ES (97.5%)", f"{res_df['ES_975'].mean():.2%}")

# 5. Visualizations
st.write("### VaR Backtesting Chart")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(res_df['Loss'], label='Actual Daily Loss', color='lightgray', alpha=0.6)
ax.plot(res_df['VaR_Param'], label='VaR Parametric (99%)', color='blue', linestyle='--')
ax.plot(res_df['VaR_Emp'], label='VaR Empirical (99%)', color='red')
ax.legend()
st.pyplot(fig)

# 6. Data Table & Download
st.write("### Detailed Calculation Table")
st.dataframe(res_df.style.format("{:.4f}"))

# Download Button for Professor
csv = res_df.to_csv().encode('utf-8')
st.download_button(" Download Results for Submission", data=csv, file_name="VaR_Project_Results.csv", mime="text/csv")