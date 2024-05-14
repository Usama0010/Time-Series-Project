from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('all_stocks_5yr.csv')

# ARIMA Model
def arima_forecast(ts):
    p, d, q = 1, 1, 1
    train_data, test_data = train_test_split(ts, test_size=0.2, shuffle=False)
    model = ARIMA(train_data, order=(p, d, q))
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(test_data))
    return predictions

# ANN Model
def ann_forecast(ts):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume']])

    def create_dataset(data, look_back=1):
        X, Y = [], []
        for i in range(len(data) - look_back - 1):
            a = data[i:(i + look_back), :]
            X.append(a)
            Y.append(data[i + look_back, 3])  # Target is 'close' price
        return np.array(X), np.array(Y)

    look_back = 3
    X, Y = create_dataset(scaled_data, look_back)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[0:train_size], X[train_size:]
    Y_train, Y_test = Y[0:train_size], Y[train_size:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

    model = Sequential()
    model.add(Dense(64, input_dim=look_back * 5, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=10, batch_size=10, verbose=1, validation_split=0.2)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(np.concatenate((np.zeros((predictions.shape[0], 4)), predictions), axis=1))[:, 4]
    return predictions

# SARIMA Model
def sarima_forecast(ts):
    train_size = int(len(ts) * 0.8)
    train, test = ts[:train_size], ts[train_size:]
    sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
    sarima_fit = sarima_model.fit(disp=False)
    sarima_forecast = sarima_fit.get_forecast(steps=len(test))
    sarima_predictions = sarima_forecast.predicted_mean
    return sarima_predictions

# ETS Model
def ets_forecast(ts):
    train_size = int(len(ts) * 0.8)
    train, test = ts[:train_size], ts[train_size:]
    model = ExponentialSmoothing(train, seasonal_periods=365, trend='add', seasonal='add', damped_trend=True)
    model_fit = model.fit(optimized=True)
    predictions = model_fit.forecast(steps=len(test))
    return predictions

# LSTM Model
def lstm_forecast(ts):
    df_for_training = df[['close']].astype(float)
    scaler = MinMaxScaler()
    df_for_training_scaled = scaler.fit_transform(df_for_training)

    def create_dataset(dataset, time_step=1):
        X, y = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            X.append(a)
            y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 100
    X, y = create_dataset(df_for_training_scaled, time_step)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[0:train_size], X[train_size:]
    y_train, y_test = y[0:train_size], y[train_size:]

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=64, verbose=1)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions).ravel()
    return predictions

# Hybrid Model
def hybrid_forecast(ts):
    train_size = int(len(ts) * 0.8)
    train_data, test_data = ts[:train_size], ts[train_size:]

    arima_model = ARIMA(train_data, order=(1, 1, 1))
    arima_fit = arima_model.fit()
    arima_pred = arima_fit.forecast(steps=len(test_data))
    arima_residuals = test_data - arima_pred

    scaler = MinMaxScaler()
    residuals_scaled = scaler.fit_transform(arima_residuals.values.reshape(-1, 1))

    ann_model = Sequential([
        Dense(50, input_dim=1, activation='relu'),
        Dense(1)
    ])
    ann_model.compile(optimizer='adam', loss='mean_squared_error')
    ann_model.fit(residuals_scaled, residuals_scaled, epochs=30, batch_size=10, verbose=1)
    ann_pred = ann_model.predict(residuals_scaled)
    ann_pred_original = scaler.inverse_transform(ann_pred)

    final_prediction = arima_pred + ann_pred_original.ravel()
    return final_prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    model_name = request.form['model']
    ts = df['close']

    if model_name == 'ARIMA':
        predictions = arima_forecast(ts)
    elif model_name == 'ANN':
        predictions = ann_forecast(ts)
    elif model_name == 'SARIMA':
        predictions = sarima_forecast(ts)
    elif model_name == 'ETS':
        predictions = ets_forecast(ts)
    elif model_name == 'LSTM':
        predictions = lstm_forecast(ts)
    elif model_name == 'Hybrid':
        predictions = hybrid_forecast(ts)
    else:
        return jsonify({'result': 'Invalid model selection'})

    # Plotting predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.plot(ts.values, label='Original')
    plt.plot(range(len(ts) - len(predictions), len(ts)), predictions, label='Predicted', color='red')
    plt.legend(loc='best')
    plt.title(f'{model_name} Model Forecast vs Actuals')
    plt.savefig('forecast_plot.png')
    plt.close()

    # Convert plot to base64
    img = BytesIO()
    with open("forecast_plot.png", "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode('utf-8')

    return jsonify({'result': 'Forecast data based on ' + model_name, 'plot': img_base64})

if __name__ == '__main__':
    app.run(debug=True)
