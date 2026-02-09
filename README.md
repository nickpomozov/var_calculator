# 🛡️ Financial Risk Management Dashboard: VaR & ES

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**An interactive risk management tool designed to calculate, visualize, and backtest Value at Risk (VaR) and Expected Shortfall (ES) for major financial indices.**

---

## 📌 Project Overview
This project was developed as part of the **Master SEP (2025-2026)** curriculum at the **Université de Reims Champagne-Ardenne**. 

The goal is to implement a robust risk engine that compares **Parametric** and **Empirical** methods to estimate potential daily losses, specifically focusing on the **CAC 40** index. The tool performs rigorous backtesting over a 1-year horizon using a 2-year rolling window (504 trading days).

### 🚀 Live Demo
**[Click here to launch the Interactive Dashboard](INSERT_YOUR_RENDER_LINK_HERE)**

---

## ⚙️ Methodology

The application implements three distinct risk metrics:

### 1. Parametric VaR (99%)
* **Assumption:** Returns follow a Normal Distribution $N(\mu, \sigma^2)$.
* **Formula:** $VaR = -(\mu + \sigma \cdot 2.326)$
* **Pros/Cons:** Fast to calculate but underestimates risk during "fat tail" events (market crashes).

### 2. Empirical VaR (99%)
* **Assumption:** No distribution assumption (Non-parametric).
* **Method:** Uses the historical 1st percentile of returns over the last 504 days.
* **Pros/Cons:** Captures actual historical crashes but reacts slowly to volatility changes (Ghost Effect).

### 3. Expected Shortfall (97.5%)
* **Definition:** The average loss *conditional* on the loss exceeding the VaR threshold.
* **Regulatory Context:** Aligned with **Basel IV (FRTB)** standards for tail risk assessment.

---

## 📂 Project Structure

```text
├── app.py                  # Main Streamlit application (Frontend)
├── calculations.py         # Calculation engine for static analysis (Backend)
├── requirements.txt        # Python dependencies
├── Financial_Risk_Project_Results.csv  # Output data for Excel analysis
└── README.md               # Project documentation
