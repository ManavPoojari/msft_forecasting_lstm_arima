# 📈 MSFT Stock Price Forecasting using LSTM and ARIMA
# Author: Manav Poojari

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ------------------- 1. Load MSFT Stock Data -------------------
df = yf.download('MSFT', start='2020-01-01', end='2025-05-31')
df = df[['Close']].dropna()

# ------------------- 2. Normalize Prices for LSTM -------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# ------------------- 3. Create Dataset for LSTM -------------------
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i - time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data)
X = X.reshape(X.shape[0], X.shape[1], 1)

# ------------------- 4. Build & Train LSTM Model -------------------
lstm_model = Sequential()
lstm_model.add(LSTM(100, input_shape=(X.shape[1], 1)))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')

# Training the model silently (verbose=0)
lstm_model.fit(X, y, epochs=20, batch_size=32, verbose=0)

# ------------------- 5. Forecast Next 6 Months using LSTM -------------------
# Predict next 126 business days (≈6 months)
input_seq = scaled_data[-60:]
lstm_preds_scaled = []

for _ in range(126):
    x = input_seq.reshape(1, 60, 1)
    pred = lstm_model.predict(x, verbose=0)[0][0]
    lstm_preds_scaled.append(pred)

    # Append predicted value and move window forward
    input_seq = np.append(input_seq[1:], [[pred]], axis=0)

# Inverse transform to get real stock prices
lstm_preds = scaler.inverse_transform(np.array(lstm_preds_scaled).reshape(-1, 1))

# ------------------- 6. ARIMA Forecast -------------------
arima_model = ARIMA(df['Close'], order=(5, 1, 0))
arima_fit = arima_model.fit()
arima_preds = arima_fit.forecast(steps=126)

# ------------------- 7. Generate Dates for Forecast -------------------
forecast_dates = pd.date_range(start='2025-06-01', periods=126, freq='B')

# ------------------- 8. Plotting Both Forecasts -------------------
plt.figure(figsize=(14, 6))
plt.plot(df['Close'][-60:], label='Recent Actual Prices', color='black')
plt.plot(forecast_dates, lstm_preds, label='LSTM Forecast', color='green')
plt.plot(forecast_dates, arima_preds, label='ARIMA Forecast', color='blue')
plt.title("MSFT Stock Forecast: LSTM vs ARIMA (Jun–Nov 2025)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------- 9. Save Results to CSV -------------------
export_df = pd.DataFrame({
    'Date': forecast_dates,
    'LSTM_Forecast': lstm_preds.flatten(),
    'ARIMA_Forecast': arima_preds.values
})
export_df.to_csv("MSFT_LSTM_ARIMA_6_Month_Forecast.csv", index=False)

print("Forecast completed.")
print("Output saved as: MSFT_LSTM_ARIMA_6_Month_Forecast.csv")
