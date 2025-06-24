#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import plotly.graph_objects as go

#######################


#######################
# Page configuration
st.set_page_config(
    page_title="Stock prediction ",
    page_icon="$$$",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")


#######################
# CSS styling
st.markdown("""
<style>

[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

</style>
""", unsafe_allow_html=True)



st.title('Stock prediction app ')

#######################

@st.cache_data
def load_all_data():
    return {
        'meta': pd.read_csv('final_meta.csv'),
        'amzn': pd.read_csv('final_amzn.csv'),
        'aapl': pd.read_csv('final_aapl.csv'),
        'nflx': pd.read_csv('final_nflx.csv'),
        'goog': pd.read_csv('final_goog.csv'),
    }

stocks = ('meta', 'amzn', 'aapl', 'nflx', 'goog')


st.title("📊 Stock Forecast Dashboard")
selected_stock = st.selectbox('Select dataset for prediction', stocks)
n_months = st.slider('Months of prediction', 1, 12)
period = n_months * 20 # Assuming ~20 trading days per month



# Loading the data 
data_load_state = st.text('Loading data...')
all_data = load_all_data()
data = all_data[selected_stock].copy()
data.reset_index(drop=True, inplace=True)
data_load_state.text('Loading data done!')

st.subheader(f"Raw data for {selected_stock.upper()}")
st.dataframe(data.tail())




def plot_raw_data(data, stock_name):
    fig = px.line(data, x="Date", y="Close", title=f"{stock_name.upper()} Closing Prices Over Time")

    # Add date range slider and buttons
    fig.update_xaxes(
        title_text="Date",
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    # Display plot in Streamlit
    st.plotly_chart(fig)

plot_raw_data(data, selected_stock)



# Forecasting SARIMAX MODELS
sarimax_params = {
    'meta': {'order': (1, 1, 2), 'trend': (0, 0), 'seasonal_order': (0, 1, 1, 20)},
    'amzn': {'order': (2, 1, 1), 'trend': (1, 1), 'seasonal_order': (1, 2, 1, 20)},
    'aapl': {'order': (1, 1, 1), 'trend': (1, 1), 'seasonal_order': (0, 1, 0, 21)},
    'nflx': {'order': (4, 1, 1), 'trend': (0, 0), 'seasonal_order': (0, 1, 0, 19)},
    'goog': {'order': (4, 0, 3), 'trend': (1, 1), 'seasonal_order': (1, 1, 0, 19)}
}
def run_sarimax_forecast(df, company_name):
    df = df.sort_values('Date').reset_index(drop=True)

    train_size = int(len(df) * 0.8)
    train = df[:train_size]
    test = df[train_size:]

    params = sarimax_params[company_name]

    model = SARIMAX(
        endog=train['Close'],
        exog=train[['sentiment_imputed']],
        order=params['order'],
        trend=params['trend'],
        seasonal_order=params['seasonal_order']
    ).fit(disp=False)

    forecast = model.forecast(
        steps=len(test),
        exog=test[['sentiment_imputed']]
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train['Date'], train['Close'], label='Train')
    ax.plot(test['Date'], test['Close'], label='Test')
    ax.plot(test['Date'], forecast, label='Forecast')
    #ax.tick_params(axis='x', rotation=45)
    ax.set_title(f"SARIMAX Forecast for {company_name.upper()}")
    ax.legend()

    st.pyplot(fig)

df_selected = all_data[selected_stock]

if st.button('Run SARIMAX Forecast'):
    run_sarimax_forecast(df_selected, selected_stock)


#########################################################
#########################################################
# Recursive forecasting 

import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor
from skforecast.recursive import ForecasterRecursive 
from skforecast.model_selection import grid_search_forecaster, TimeSeriesFold
import skforecast


def run_recursive_forecast(df, company_name):
    df = df.sort_values('Date').reset_index(drop=True)

    # Ensure datetime format and set index if needed
    df['Date'] = pd.to_datetime(df['Date'])

    steps = 51
    train = df[:-steps]
    test = df[-steps:]

    # Create and train the forecaster
    forecaster = ForecasterRecursive(
        regressor=DecisionTreeRegressor(random_state=123),
        lags=7,
        differentiation=1
    )
    forecaster.fit(y=train['Close'], store_in_sample_residuals=True)

    # Generate predictions
    predictions = forecaster.predict(steps=steps)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train['Date'], train['Close'], label='Train')
    ax.plot(test['Date'], test['Close'], label='Test')
    ax.plot(test['Date'], predictions, label='Forecast')
    ax.set_title(f"Recursive Forecast (Decision Tree) for {company_name.upper()}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.legend()
    ax.tick_params(axis='x', rotation=45)

    # Show plot in Streamlit
    st.pyplot(fig)

df_selected = all_data[selected_stock]  # e.g., {'meta': final_meta, ...}

if st.button('Run Recursive Forecast'):
    run_recursive_forecast(df_selected, selected_stock)





### Deep learning 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

def run_lstm_forecast(df, stock_name):
    df = df.copy()

    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Features and target
    features = df[['Open', 'High', 'Low', 'Close', 'sentiment_imputed']]
    target = df['Close']

    # Scaling
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Prepare sequences
    n_past = 14
    n_future = 1
    trainX, trainY = [], []

    for i in range(n_past, len(features_scaled) - n_future + 1):
        trainX.append(features_scaled[i - n_past:i])
        trainY.append(features_scaled[i:i + n_future, 0])  # Predict 'Open'

    trainX, trainY = np.array(trainX), np.array(trainY)

    # Model
    model = Sequential([
        LSTM(32, activation='relu', return_sequences=True, input_shape=(n_past, features.shape[1])),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train
    model.fit(trainX, trainY, epochs=50, validation_split=0.1, verbose=0)

    # Forecast next 7 days
    n_future = 7
    forecast_dates = pd.date_range(df['Date'].iloc[-1], periods=n_future + 1, freq='1D')[1:]
    forecast_input = trainX[-n_future:]
    forecast = model.predict(forecast_input)

    # Inverse transform
    repeated_forecast = np.repeat(forecast, features.shape[1], axis=-1)
    forecast_prices = scaler.inverse_transform(repeated_forecast)[:, 0]

    # Create forecast DataFrame
    df_forecast = pd.DataFrame({'Date': forecast_dates, 'Close': forecast_prices})

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['Date'], df['Close'], label='Historical', color='blue')
    ax.plot(df_forecast['Date'], df_forecast['Close'], label='Forecast', color='orange', linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.set_title(f'{stock_name} - LSTM Forecast')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Optionally show forecast table
    st.subheader("7-Day Forecasted Close Prices")
    st.dataframe(df_forecast)


df_selected = all_data[selected_stock]

if st.button('Run LSTM Forecast'):
    run_lstm_forecast(df_selected, selected_stock)



#displaying the actual stock price prediction 

# Load the CSV
df = pd.read_csv('df.csv')

# Define the correct column names based on selected stock
lstm_col = f"{selected_stock}_lstm"
recursive_col = f"{selected_stock}_recursive"

# Select rows for days 1, 3, and 7
selected_rows = df[[lstm_col, recursive_col]].iloc[[0, 2, 6]]

# Display the results
st.subheader(f"📈 Forecast (LSTM) vs Recursive for {selected_stock.upper()} - Days 1, 3, 7")
st.dataframe(selected_rows)
