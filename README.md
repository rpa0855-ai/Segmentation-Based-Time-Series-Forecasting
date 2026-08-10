# Segmentation-Based Time Series Forecasting Using Change Point Detection and Statistical Modeling

**Author:** Pradeep Kumar Yadav, M.Sc. Statistics, Indian Institute of Technology (IIT) Bombay
**Guide:** Prof. Ashok Jaiswal, MIT World Peace University (MIT-WPU), Pune
**Type:** Research Summer Internship Project

## What this project does

Normal forecasting models like ARIMA and SARIMA assume a time series behaves the same
way across its whole history. But real series often go through structural breaks -
like a financial crisis or a pandemic - after which the series behaves differently.

This project:
1. Detects structural breaks in a real economic time series (US Industrial Production
   Index, 1995-2026) using two methods - **PELT** and **Binary Segmentation**
2. Splits the series into 9 homogeneous segments (both methods agree, and both
   correctly identify the 2008 financial crisis and the 2020 COVID-19 crash)
3. Fits **ARIMA/SARIMA** models to each segment separately, selected using AIC/BIC
4. Checks residuals using the **Ljung-Box test**
5. Builds a **weighted ensemble** forecast that favors recent segments
6. Validates everything using rolling-origin and expanding-window backtests
7. Tests whether any accuracy difference is statistically real using the
   **Diebold-Mariano test**, not just comparing numbers

## Honest result

The segment-wise model (using only the most recent regime) performs about the same
as a single, well-built model - no statistically significant difference. But
combining *all* segments into one big weighted ensemble actually performed
significantly *worse* than a single model, confirmed by the Diebold-Mariano test.
Change-point detection itself worked very well and picked out real, well-known
economic events correctly. This project reports that honestly rather than only
showing the results that support the original idea.

## Repository structure

```
├── notebook/
│   └── Segmentation_Based_Time_Series_Forecasting...ipynb   <- full analysis, runs in Google Colab
├── report/
│   └── Segmentation_Based_Forecasting_Full_Report.pdf        <- 47-page written report
├── streamlit_app/
│   ├── app.py              <- interactive web app
│   └── requirements.txt
└── README.md
```

## How to run the notebook

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `notebook/Segmentation_Based_Time_Series_Forecasting...ipynb`
3. Runtime → Run all (data downloads automatically from FRED, no upload needed)

## How to run the interactive app

Live demo: *(add your Streamlit Cloud URL here after deploying)*

To run locally:
```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

## Methods used

- **Change point detection:** PELT, Binary Segmentation (`ruptures`)
- **Stationarity testing:** Augmented Dickey-Fuller (ADF) test
- **Forecasting models:** ARIMA, SARIMA (`statsmodels`)
- **Model selection:** AIC, BIC
- **Residual diagnostics:** Ljung-Box test, Q-Q plots
- **Ensemble:** recency- and reliability-weighted forecast combination
- **Validation:** single hold-out, rolling forecast origin, expanding window backtest
- **Significance testing:** Diebold-Mariano test

## Tech stack

Python, Pandas, NumPy, Matplotlib, Statsmodels, Ruptures, SciPy, Scikit-learn, Streamlit
