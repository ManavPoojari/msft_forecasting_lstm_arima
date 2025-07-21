# 📈 MSFT Stock Forecasting – LSTM vs ARIMA

Hi! This project is about forecasting Microsoft (MSFT) stock prices using two different time series models: LSTM (deep learning) and ARIMA (statistical). I wanted to explore how traditional and modern techniques compare when it comes to real-world stock data.


## 🧰 Tools & Libraries Used

- Python, pandas, NumPy, matplotlib
- TensorFlow/Keras for LSTM model
- statsmodels for ARIMA
- yFinance for stock data
- Power BI for dashboarding

---

## 🧪 What I Did

- Pulled 5 years of MSFT stock data using `yfinance`
- Built and trained an LSTM model for sequence prediction
- Used ARIMA for classical time series forecasting
- Compared both forecasts for the next 6 months (Jun–Nov 2025)
- Visualized results using both Python (matplotlib) and Power BI

---

## 📁 Project Structure
msft-forecasting-lstm-arima/
├── data/
│ └── MSFT_LSTM_ARIMA_6_Month_Forecast.csv
│
├── images/
│ ├── Output.png
│ └── powerbi_forecast_chart.png
│
├── msft_forecasting_lstm_arima.py
├── README.md

🧠 Why LSTM vs ARIMA?
Model	Good For	Notes
LSTM	Nonlinear, long-term patterns	Needs more data and compute
ARIMA	Simpler trends	Quick to train and interpret

I wanted to compare them side by side on real stock data.

📬 About Me
Manav Poojari
📍 Mumbai, India
🎓 MSc IT Graduate | 🎓 MSc IT Graduate | Passionate about data analysis, predictive modeling, and real-world problem solving
📧 manavpoojari24@gmail.com
https://www.linkedin.com/in/manav-poojari-b3579a213

Thanks for checking this out! 😊


