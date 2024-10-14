from flask import Flask, request, jsonify, send_from_directory
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import logging
from flask_cors import CORS
from datetime import datetime
import os 

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# Function to fetch stock data
enddt = datetime.now().strftime('%Y-%m-%d')
def get_stock_data(symbol, start='2010-01-01', end=enddt):
    logging.info(f"Fetching stock data for symbol: {symbol}")
    stock_data = yf.download(symbol, start=start, end=end)
    
    if stock_data.empty:
        raise ValueError("No data found for the provided stock symbol.")
    
    stock_data = stock_data[['Close']]  # Only keep the 'Close' column
    stock_data = stock_data.ffill()  # Fill missing data (forward fill)
    logging.info(f"Stock data fetched successfully. Data size: {len(stock_data)} rows")
    return stock_data

# Function to train the model and predict the stock price
def train_model(stock_data, prediction_days=15):
    logging.info(f"Training model with prediction days: {prediction_days}")
    
    # Create the 'Prediction' column (next day's close price)
    stock_data['Prediction'] = stock_data[['Close']].shift(-prediction_days)
    
    # Define features (X) and labels (y)
    X = stock_data.drop(['Prediction'], axis=1).values[:-prediction_days]  # Closing price as features
    y = stock_data['Prediction'].values[:-prediction_days]  # Next day's closing price as labels

    # If there isn't enough data to make a prediction
    if len(X) < prediction_days:
        raise ValueError("Not enough data to train the model.")
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Now we predict the next day's stock price using the last available close price (1 feature)
    last_close_price = stock_data['Close'].values[-1].reshape(-1, 1)  # Reshape for single feature
    future_price = model.predict(last_close_price)

    logging.info(f"Prediction completed. Future price: {future_price[0]}")
    
    return future_price[0]

# Route for the homepage
@app.route('/')
def home():
    return send_from_directory(os.getcwd(), 'home.html') # type: ignore

INDIAN_STOCKS = {
    "HINDALCO": "HINDALCO.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "RELIANCE": "RELIANCE.NS",
    "INFY": "INFY.NS",
    "TCS": "TCS.NS",
    "HDFC": "HDFC.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "LT": "LT.NS",
    # Add more stocks as needed
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logging.info("Received a request for prediction")
        data = request.get_json()
        symbol = data.get('symbol', '').strip().upper()
        
        if not symbol:
            logging.warning("No stock symbol provided")
            return jsonify({'error': 'Please provide a valid stock symbol.'}), 400

        # Check if the symbol is in the Indian stocks dictionary
        if symbol in INDIAN_STOCKS:
            symbol = INDIAN_STOCKS[symbol]  # Use the mapped symbol
        
        logging.info(f"Stock symbol received: {symbol}")
        
        # Fetch stock data and make the prediction
        stock_data = get_stock_data(symbol)
        logging.info(f"Stock data fetched for {symbol}")
        
        future_price = train_model(stock_data)
        logging.info(f"Prediction made for {symbol}: {future_price:.2f}")

        # Determine currency based on symbol
        if symbol.endswith('.NS'):
            currency = '₹'  # Indian Rupee for NSE stocks
        else:
            currency = '$'  # US Dollar for other stocks

        return jsonify({'prediction': f"The predicted price for {symbol} is {currency}{future_price:.2f}"})
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5500)
