import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import datetime

# Load NSE symbols for dropdown
@st.cache_data
def get_nse_symbols():
    nse_symbols_df = pd.read_csv("nse_symbols.csv")  # Load NSE symbols from CSV
    return nse_symbols_df['Symbol'].tolist()

# Load stock data for the given ticker
def load_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            st.error("No data found for the entered ticker. Please check the symbol and try again.")
            return None
        return stock_data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

# Train LSTM model
def train_lstm_model(stock_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))
    
    X_train, y_train = [], []
    look_back = 60
    
    for i in range(look_back, len(scaled_data)):
        X_train.append(scaled_data[i-look_back:i, 0])
        y_train.append(scaled_data[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
    
    return model, scaler

# Predict future prices
def predict_future_prices(stock_data, model, scaler, future_days=30):
    scaled_data = scaler.transform(stock_data['Close'].values.reshape(-1, 1))
    look_back = 60
    
    last_60_days = scaled_data[-look_back:]
    X_future = np.reshape(last_60_days, (1, look_back, 1))
    predictions = []
    
    for _ in range(future_days):
        pred_price = model.predict(X_future)[0][0]
        predictions.append(pred_price)
        
        # Correctly reshape and append predicted value
        X_future = np.append(X_future[:, 1:, :], np.array(pred_price).reshape(1, 1, 1), axis=1)

    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predicted_prices

# Main Streamlit app
def main():
    st.title("Stock Price Prediction App 📈")

    # Load NSE symbols
    nse_symbols = get_nse_symbols()

    # Stock selection dropdown
    ticker = st.selectbox(
        "Choose a Stock Ticker from NSE or Enter Custom Symbol Below:",
        nse_symbols,
        index=nse_symbols.index("RELIANCE.NS") if "RELIANCE.NS" in nse_symbols else 0
    )

    # Text input for custom ticker
    custom_ticker = st.text_input("Or Enter Custom Stock Ticker (e.g., WIPRO.NS)", "")
    if custom_ticker:
        ticker = custom_ticker

    # Select date range for data
    start_date = st.date_input("Select Start Date for Data:", datetime.date(2020, 1, 1))
    end_date = st.date_input("Select End Date for Data:", datetime.date.today())
    
    # Load stock data
    if st.button("Load Stock Data"):
        stock_data = load_stock_data(ticker, start_date, end_date)
        if stock_data is not None:
            st.write(f"## {ticker} Stock Data")
            st.write(stock_data.tail())
            
            st.write("### Stock Closing Price Over Time")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(stock_data['Close'], label='Closing Price', color='blue')
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)
    
    # Train LSTM model
    if st.button("Train LSTM Model"):
        stock_data = load_stock_data(ticker, start_date, end_date)
        if stock_data is not None:
            model, scaler = train_lstm_model(stock_data)
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
            st.success("✅ Model trained successfully!")

    # Predict future prices
    if st.button("Predict Future Prices"):
        if 'model' in st.session_state and 'scaler' in st.session_state:
            stock_data = load_stock_data(ticker, start_date, end_date)
            
            if stock_data is not None:
                model = st.session_state['model']
                scaler = st.session_state['scaler']
                future_prices = predict_future_prices(stock_data, model, scaler)
                
                st.write("### Future Price Predictions")
                future_dates = pd.date_range(start=stock_data.index[-1], periods=31, freq='D')[1:]
                future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_prices.flatten()})
                st.write(future_df)
                
                # Plot future predictions
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(stock_data.index, stock_data['Close'], label=f'{ticker} Historical Prices', color='blue')
                ax.plot(future_dates, future_prices, label='Predicted Future Prices', color='red', linestyle='dashed')
                ax.set_xlabel("Date")
                ax.set_ylabel("Stock Price")
                ax.legend()
                st.pyplot(fig)
            else:
                st.warning("⚠ Invalid stock symbol. Please try again.")
        else:
            st.warning("⚠ Please train the model first!")
    
    # Display company details
    st.write("### Company Details")
    stock_info = yf.Ticker(ticker).info
    st.write(f"*Company Name:* {stock_info.get('longName', 'N/A')}")
    st.write(f"*Sector:* {stock_info.get('sector', 'N/A')}")
    st.write(f"*Industry:* {stock_info.get('industry', 'N/A')}")
    st.write(f"*Market Cap:* {stock_info.get('marketCap', 'N/A')}")
    st.write(f"*Website:* [Click Here]({stock_info.get('website', '#')})")
    st.write(f"*Description:* {stock_info.get('longBusinessSummary', 'N/A')}")

if _name_ == "_main_":
    main()