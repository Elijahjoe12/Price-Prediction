from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

# Set Matplotlib to non-interactive backend
matplotlib.use('Agg')

app = Flask(__name__)

# Load Pre-trained Model
model = load_model("btc.keras")

# Helper Function to Convert Matplotlib Plots to HTML
def plot_to_html(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    buf.close()
    return f"data:image/png;base64,{data}"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        coin = request.form.get("coin")
        no_of_days = int(request.form.get("no_of_days"))
        return redirect(url_for("predict", coin=coin, no_of_days=no_of_days))
    return render_template("index.html")

@app.route("/predict")
def predict():
    coin = request.args.get("coin", "BTC-USD")
    no_of_days = int(request.args.get("no_of_days", 10))

    # Fetch Stock Data
    end = datetime.now()
    start = datetime(end.year - 7, end.month, end.day)
    coin_data = yf.download(coin, start, end)
    if coin_data.empty:
        return render_template("result.html", error="Invalid coin ticker or no data available.")

    # Data Preparation
    splitting_len = int(len(coin_data) * 0.9)
    x_test = coin_data[['Close']][splitting_len:]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(x_test)

    x_data = []
    y_data = []
    for i in range(100, len(scaled_data)):
        x_data.append(scaled_data[i - 100:i])
        y_data.append(scaled_data[i])

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Predictions
    predict = model.predict(x_data)
    inv_predict = scaler.inverse_transform(predict)
    inv_ytest = scaler.inverse_transform(y_data)

    # Prepare Data for Plotting
    plot_data = pd.DataFrame({
        'Original Test Data': inv_ytest.flatten(),
        'Predicted Test Data': inv_predict.flatten()
    }, index=x_test.index[100:])

    # Generate Plots
    # Plot 1: Original Closing Prices
    fig1 = plt.figure(figsize=(15, 6))
    plt.plot(coin_data['Close'], 'b', label='Close Price')
    plt.title("Closing Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.legend()
    original_plot = plot_to_html(fig1)

    # Plot 2: Original vs Predicted Test Data
    fig2 = plt.figure(figsize=(15, 6))
    plt.plot(plot_data['Original Test Data'], label="Original Test Data")
    plt.plot(plot_data['Predicted Test Data'], label="Predicted Test Data", linestyle="--")
    plt.legend()
    plt.title("Original vs Predicted Closing Prices")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    predicted_plot = plot_to_html(fig2)

    # Plot 3: Future Predictions
    last_100 = coin_data[['Close']].tail(100)
    last_100_scaled = scaler.transform(last_100)

    fut_predict = []
    last_100_scaled = last_100_scaled.reshape(1, -1, 1)
    for _ in range(no_of_days):
        next_day = model.predict(last_100_scaled)
        fut_predict.append(scaler.inverse_transform(next_day))
        last_100_scaled = np.append(last_100_scaled[:, 1:, :], next_day.reshape(1, 1, -1), axis=1)

    fut_predict = np.array(fut_predict).flatten()

    fig3 = plt.figure(figsize=(15, 6))
    plt.plot(range(1, no_of_days + 1), fut_predict, marker='o', label="Predicted Future Prices", color="purple")
    plt.title("Future Closing Price Predictions")
    plt.xlabel("Days Ahead")
    plt.ylabel("Predicted Closing Price")
    plt.grid(alpha=0.3)
    plt.legend()
    future_plot = plot_to_html(fig3)

    return render_template(
        "result.html",
        coin=coin,
        original_plot=original_plot,
        predicted_plot=predicted_plot,
        future_plot=future_plot,
        enumerate =enumerate,
        fut_predict=fut_predict
    )

if __name__ == "__main__":
    app.run(debug=True)