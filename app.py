import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.model_selection import ParameterGrid
import tensorflow as tf

#  CPU threads TensorFlow
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

# Loading data for Twitter Sentiment Analysis
@st.cache_data
def load_twitter_data():
    df_twitter = pd.read_csv('StreamlitData1.csv')
    df_twitter['date'] = pd.to_datetime(df_twitter['date'])
    df_twitter.set_index('date', inplace=True)
    return df_twitter

# Loading data for LSTM Time Series
@st.cache_data
def load_lstm_data():
    hsa3 = pd.read_csv('StreamlitData2.csv')
    hsa3['date'] = pd.to_datetime(hsa3['date'])
    hsa3.set_index('date', inplace=True)
    hsa3.index.freq = pd.infer_freq(hsa3.index)
    return hsa3

df_twitter = load_twitter_data()
hsa3 = load_lstm_data()

# Function to create sequences
def create_sequences(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Function to make future predictions
def predict_future(model, data_scaled, time_step, future_steps, scaler):
    temp_input = list(data_scaled[-time_step:].reshape(1, -1)[0])
    lst_output = []
    for i in range(future_steps):
        if len(temp_input) > time_step:
            x_input = np.array(temp_input[-time_step:])
            x_input = x_input.reshape((1, time_step, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.append(yhat[0][0])
            temp_input = temp_input[-time_step:]
            lst_output.append(yhat[0][0])
        else:
            x_input = np.array(temp_input).reshape((1, time_step, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.append(yhat[0][0])
            lst_output.append(yhat[0][0])
    return scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))

# Function to plot LSTM results
def plot_lstm_results(train, test, test_index, predictions, rmse):
    plt.figure(figsize=(15, 7))
    plt.plot(train.index, train, label='Train data')
    plt.plot(test.index, test, label='Test data', color='orange')
    plt.plot(test_index, predictions, label='Test Predictions', color='red')
    plt.title(f'LSTM Model - Time Series Prediction\nRMSE: {rmse:.4f}')
    plt.xlabel('Date')
    plt.ylabel('Sentiment')
    plt.legend()
    st.pyplot(plt)

# Function to plot LSTM forecasts
def plot_lstm_forecasts(train, test, test_index, predictions, model, test_scaled, time_step, scaler):
    forecast_days = [1, 3, 7]
    n_future_list = [24 * day for day in forecast_days]

    plt.figure(figsize=(15, 21))

    for i, n_future in enumerate(n_future_list):
        future_predictions = predict_future(model, test_scaled, time_step, n_future, scaler)
        future_index = pd.date_range(start=test.index[-1] + pd.Timedelta(hours=1), periods=n_future, freq='h')
        
        # Calculating RMSE for future predictions
        future_test_segment = test[-n_future:]
        if len(future_test_segment) == len(future_predictions):
            rmse_future = np.sqrt(mean_squared_error(future_test_segment, future_predictions))
        else:
            rmse_future = np.nan  # Handling cases where lengths do not match

        plt.subplot(3, 1, i+1)
        plt.plot(train.index, train, label='Train data')
        plt.plot(test.index, test, label='Test data', color='orange')
        plt.plot(test_index, predictions, label='Test Predictions', color='red')
        plt.plot(future_index, future_predictions, label=f'Future Predictions ({forecast_days[i]} day{"s" if forecast_days[i] > 1 else ""})', color='green')
        plt.title(f'LSTM Model - {forecast_days[i]} Day{"s" if forecast_days[i] > 1 else ""} Sentiment Prediction\nRMSE: {rmse_future:.4f}')
        plt.xlabel('Date')
        plt.ylabel('Sentiment')
        plt.legend()

    plt.tight_layout()
    st.pyplot(plt)


# Page 1: YCSB Workloads
def ycsb_workloads_page():
    st.title("YCSB Workloads")
    # Plotting YCSB workloads visualization code here
    def plot_workload_a_read():
        MySQL_rl = [482.01, 196.57, 262.01, 164.47, 194.01]
        MySQL_rj = [523, 5014, 25133, 49965, 100153]
        MongoDB_rl = [491.08, 123.38, 75.67, 72.13, 75.29]
        MongoDB_rj = [502, 4978, 24854, 50036, 100010]
        Cassandra_rl = [560.75, 291.58, 284.48, 264.13, 301.31]
        Cassandra_rj = [486, 4996, 25305, 49782, 99742]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=MySQL_rj, y=MySQL_rl, mode='lines+markers', name='MySQL', marker=dict(symbol="x", size=10), line=dict(color='black')))
        fig.add_trace(go.Scatter(x=Cassandra_rj, y=Cassandra_rl, mode='lines+markers', name='Cassandra', marker=dict(symbol="square", size=10), line=dict(color='gray')))
        fig.add_trace(go.Scatter(x=MongoDB_rj, y=MongoDB_rl, mode='lines+markers', name='MongoDB', marker=dict(symbol="circle", size=10), line=dict(color='lightgray')))

        fig.update_layout(
            title='Workload A: Read Latency',
            xaxis_title='Operations',
            yaxis_title='Read latency (μs)',
            legend_title='Database',
            template='simple_white'
        )
        return fig

    def plot_workload_a_update():
        MySQL_wl = [3349.12, 2588.04, 3468.35, 2797.94, 3099.49]
        MySQL_wj = [477, 4986, 24867, 50035, 99847]
        MongoDB_wl = [316.39, 153.92, 96.06, 81.77, 87.76]
        MongoDB_wj = [498, 5022, 25146, 49964, 99990]
        Cassandra_wl = [581.33, 275.06, 248.83, 234.91, 261.75]
        Cassandra_wj = [514, 5004, 24695, 50218, 100258]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=MySQL_wj, y=MySQL_wl, mode='lines+markers', name='MySQL', marker=dict(symbol="x", size=10), line=dict(color='black')))
        fig.add_trace(go.Scatter(x=Cassandra_wj, y=Cassandra_wl, mode='lines+markers', name='Cassandra', marker=dict(symbol="square", size=10), line=dict(color='gray')))
        fig.add_trace(go.Scatter(x=MongoDB_wj, y=MongoDB_wl, mode='lines+markers', name='MongoDB', marker=dict(symbol="circle", size=10), line=dict(color='lightgray')))

        fig.update_layout(
            title='Workload A: Update Latency',
            xaxis_title='Operations',
            yaxis_title='Update latency (μs)',
            legend_title='Database',
            template='simple_white'
        )
        return fig

    def plot_workload_b_read():
        MySQL_rl = [329.82, 163.48, 134.78, 124.48, 149.52]
        MySQL_rj = [945, 9506, 47505, 94960, 189854]
        MongoDB_rl = [339.67, 104.90, 71.62, 65.81, 64.77]
        MongoDB_rj = [952, 9494, 47456, 95081, 189980]
        Cassandra_rl = [805.32, 327.07, 304.04, 238.70, 298.21]
        Cassandra_rj = [953, 9531, 47488, 95092, 189899]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=MySQL_rj, y=MySQL_rl, mode='lines+markers', name='MySQL', marker=dict(symbol="x", size=10), line=dict(color='black')))
        fig.add_trace(go.Scatter(x=Cassandra_rj, y=Cassandra_rl, mode='lines+markers', name='Cassandra', marker=dict(symbol="square", size=10), line=dict(color='gray')))
        fig.add_trace(go.Scatter(x=MongoDB_rj, y=MongoDB_rl, mode='lines+markers', name='MongoDB', marker=dict(symbol="circle", size=10), line=dict(color='lightgray')))

        fig.update_layout(
            title='Workload B: Read Latency',
            xaxis_title='Operations',
            yaxis_title='Read latency (μs)',
            legend_title='Database',
            template='simple_white'
        )
        return fig

    def plot_workload_b_update():
        MySQL_wl = [4932.49, 3438.93, 3031.89, 2901.64, 2989.87]
        MySQL_wj = [55, 494, 2495, 5040, 10146]
        MongoDB_wl = [500.45, 252.04, 138.18, 122.83, 108.56]
        MongoDB_wj = [48, 506, 2544, 4919, 10020]
        Cassandra_wl = [1374.44, 558.94, 379.50, 273.15, 307.05]
        Cassandra_wj = [47, 469, 2512, 4908, 10101]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=MySQL_wj, y=MySQL_wl, mode='lines+markers', name='MySQL', marker=dict(symbol="x", size=10), line=dict(color='black')))
        fig.add_trace(go.Scatter(x=Cassandra_wj, y=Cassandra_wl, mode='lines+markers', name='Cassandra', marker=dict(symbol="square", size=10), line=dict(color='gray')))
        fig.add_trace(go.Scatter(x=MongoDB_wj, y=MongoDB_wl, mode='lines+markers', name='MongoDB', marker=dict(symbol="circle", size=10), line=dict(color='lightgray')))

        fig.update_layout(
            title='Workload B: Update Latency',
            xaxis_title='Operations',
            yaxis_title='Update latency (μs)',
            legend_title='Database',
            template='simple_white'
        )
        return fig

    def plot_workload_c_read():
        MySQL_rl = [429.40, 136.42, 106.78, 110.37, 121.04]
        MySQL_rj = [1000, 10000, 50000, 100000, 200000]
        MongoDB_rl = [323.70, 103.89, 60.82, 69.95, 59.82]
        MongoDB_rj = [1000, 10000, 50000, 100000, 200000]
        Cassandra_rl = [509.33, 349.11, 256.92, 307.42, 318.49]
        Cassandra_rj = [1000, 10000, 50000, 100000, 200000]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=MySQL_rj, y=MySQL_rl, mode='lines+markers', name='MySQL', marker=dict(symbol="x", size=10), line=dict(color='black')))
        fig.add_trace(go.Scatter(x=Cassandra_rj, y=Cassandra_rl, mode='lines+markers', name='Cassandra', marker=dict(symbol="square", size=10), line=dict(color='gray')))
        fig.add_trace(go.Scatter(x=MongoDB_rj, y=MongoDB_rl, mode='lines+markers', name='MongoDB', marker=dict(symbol="circle", size=10), line=dict(color='lightgray')))

        fig.update_layout(
            title='Workload C: Read Latency',
            xaxis_title='Operations',
            yaxis_title='Read latency (μs)',
            legend_title='Database',
            template='simple_white'
        )
        return fig

    def plot_workload_c_overall():
        MySQL_or = [813, 1766, 5828, 11575, 24919]
        MySQL_ot = [1230.01, 5662.51, 8579.24, 8639.31, 8026.00]
        MongoDB_or = [614, 1433, 3544, 7574, 12620]
        MongoDB_ot = [1628.66, 6978.36, 14108.35, 13203.06, 15847.86]
        Cassandra_or = [3239, 6272, 15685, 33850, 66902]
        Cassandra_ot = [308.74, 1594.39, 3187.76, 2954.21, 2989.45]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=MySQL_ot, y=MySQL_or, mode='lines+markers', name='MySQL', marker=dict(symbol="x", size=10), line=dict(color='black')))
        fig.add_trace(go.Scatter(x=Cassandra_ot, y=Cassandra_or, mode='lines+markers', name='Cassandra', marker=dict(symbol="square", size=10), line=dict(color='gray')))
        fig.add_trace(go.Scatter(x=MongoDB_ot, y=MongoDB_or, mode='lines+markers', name='MongoDB', marker=dict(symbol="circle", size=10), line=dict(color='lightgray')))

        fig.update_layout(
            title='Workload C: Overall',
            xaxis_title='Throughput (ops/sec)',
            yaxis_title='Runtime (ms)',
            legend_title='Database',
            template='simple_white'
        )
        return fig

    def plot_workload_d_read():
        MySQL_rl = [420.03, 162.50, 126.83, 142.54, 142.41]
        MySQL_rj = [948, 9524, 47530, 94934, 189944]
        MongoDB_rl = [304.27, 112.01, 65.83, 59.33, 60.77]
        MongoDB_rj = [937, 9507, 47529, 94880, 190016]
        Cassandra_rl = [1058.97, 436.89, 270.57, 352.93, 391.99]
        Cassandra_rj = [950, 9520, 47540, 95053, 189939]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=MySQL_rj, y=MySQL_rl, mode='lines+markers', name='MySQL', marker=dict(symbol="x", size=10), line=dict(color='black')))
        fig.add_trace(go.Scatter(x=Cassandra_rj, y=Cassandra_rl, mode='lines+markers', name='Cassandra', marker=dict(symbol="square", size=10), line=dict(color='gray')))
        fig.add_trace(go.Scatter(x=MongoDB_rj, y=MongoDB_rl, mode='lines+markers', name='MongoDB', marker=dict(symbol="circle", size=10), line=dict(color='lightgray')))

        fig.update_layout(
            title='Workload D: Read Latency',
            xaxis_title='Operations',
            yaxis_title='Read latency (μs)',
            legend_title='Database',
            template='simple_white'
        )
        return fig

    def plot_workload_d_insert():
        MySQL_il = [4002.00, 3613.01, 3048.79, 3006.92, 3041.19]
        MySQL_ij = [52, 476, 2470, 5066, 10056]
        MongoDB_il = [389.08, 213.06, 116.06, 101.05, 89.78]
        MongoDB_ij = [63, 493, 2471, 5120, 9984]
        Cassandra_il = [1420.82, 631.50, 324.96, 451.48, 458.92]
        Cassandra_ij = [50, 480, 2460, 4947, 10061]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=MySQL_ij, y=MySQL_il, mode='lines+markers', name='MySQL', marker=dict(symbol="x", size=10), line=dict(color='black')))
        fig.add_trace(go.Scatter(x=Cassandra_ij, y=Cassandra_il, mode='lines+markers', name='Cassandra', marker=dict(symbol="square", size=10), line=dict(color='gray')))
        fig.add_trace(go.Scatter(x=MongoDB_ij, y=MongoDB_il, mode='lines+markers', name='MongoDB', marker=dict(symbol="circle", size=10), line=dict(color='lightgray')))

        fig.update_layout(
            title='Workload D: Insert Latency',
            xaxis_title='Operations',
            yaxis_title='Insert latency (μs)',
            legend_title='Database',
            template='simple_white'
        )
        return fig

    def plot_workload_e_scan():
        MySQL_sl = [814.63, 527.70, 551.95, 544.17, 591.98]
        MySQL_sj = [625, 6530, 32772, 65424, 130915]
        MongoDB_sl = [502.34, 334.07, 258.67, 287.90, 322.51]
        MongoDB_sj = [645, 6465, 32726, 65246, 131418]
        Cassandra_sl = [1461.79, 898.25, 1027.27, 840.32, 824.64]
        Cassandra_sj = [664, 6550, 32905, 65821, 131147]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=MySQL_sj, y=MySQL_sl, mode='lines+markers', name='MySQL', marker=dict(symbol="x", size=10), line=dict(color='black')))
        fig.add_trace(go.Scatter(x=Cassandra_sj, y=Cassandra_sl, mode='lines+markers', name='Cassandra', marker=dict(symbol="square", size=10), line=dict(color='gray')))
        fig.add_trace(go.Scatter(x=MongoDB_sj, y=MongoDB_sl, mode='lines+markers', name='MongoDB', marker=dict(symbol="circle", size=10), line=dict(color='lightgray')))

        fig.update_layout(
            title='Workload E: Scan Latency',
            xaxis_title='Operations',
            yaxis_title='Scan latency (μs)',
            legend_title='Database',
            template='simple_white'
        )
        return fig

    def plot_workload_e_insert():
        MySQL_il = [3232.33, 2820.99, 2903.03, 3125.16, 3070.03]
        MySQL_ij = [375, 3470, 17228, 34576, 69085]
        MongoDB_il = [503.23, 175.53, 98.68, 102.10, 113.95]
        MongoDB_ij = [348, 3505, 17266, 34745, 68571]
        Cassandra_il = [1157.75, 573.68, 701.69, 438.11, 412.51]
        Cassandra_ij = [336, 3450, 17095, 34179, 68853]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=MySQL_ij, y=MySQL_il, mode='lines+markers', name='MySQL', marker=dict(symbol="x", size=10), line=dict(color='black')))
        fig.add_trace(go.Scatter(x=Cassandra_ij, y=Cassandra_il, mode='lines+markers', name='Cassandra', marker=dict(symbol="square", size=10), line=dict(color='gray')))
        fig.add_trace(go.Scatter(x=MongoDB_ij, y=MongoDB_il, mode='lines+markers', name='MongoDB', marker=dict(symbol="circle", size=10), line=dict(color='lightgray')))

        fig.update_layout(
            title='Workload E: Insert Latency',
            xaxis_title='Operations',
            yaxis_title='Insert latency (μs)',
            legend_title='Database',
            template='simple_white'
        )
        return fig

    def plot_workload_f_read_modify_write():
        MySQL_rmwl = [3284.21, 2879.39, 3199.94, 3047.24, 3079.58]
        MySQL_rmwj = [520, 4929, 25002, 50107, 99663]
        MongoDB_rmwl = [628.78, 215.63, 143.58, 153.59, 150.17]
        MongoDB_rmwj = [501, 4939, 25217, 49837, 99860]
        Cassandra_rmwl = [1493.12, 837.31, 826.31, 617.49, 653.32]
        Cassandra_rmwj = [512, 4940, 25065, 49859, 99637]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=MySQL_rmwj, y=MySQL_rmwl, mode='lines+markers', name='MySQL', marker=dict(symbol="x", size=10), line=dict(color='black')))
        fig.add_trace(go.Scatter(x=Cassandra_rmwj, y=Cassandra_rmwl, mode='lines+markers', name='Cassandra', marker=dict(symbol="square", size=10), line=dict(color='gray')))
        fig.add_trace(go.Scatter(x=MongoDB_rmwj, y=MongoDB_rmwl, mode='lines+markers', name='MongoDB', marker=dict(symbol="circle", size=10), line=dict(color='lightgray')))

        fig.update_layout(
            title='Workload F: Read-modify-write Latency',
            xaxis_title='Operations',
            yaxis_title='Read-modify-write latency (μs)',
            legend_title='Database',
            template='simple_white'
        )
        return fig

    def plot_workload_f_update():
        MySQL_ul = [2942.92, 2703.66, 3008.73, 2877.79, 2906.46]
        MySQL_uj = [520, 4929, 25002, 50107, 99663]
        MongoDB_ul = [268.65, 114.19, 77.96, 82.00, 79.47]
        MongoDB_uj = [501, 4939, 25217, 49837, 99860]
        Cassandra_ul = [806.44, 433.47, 384.78, 294.55, 312.25]
        Cassandra_uj = [512, 4940, 25065, 49859, 99637]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=MySQL_uj, y=MySQL_ul, mode='lines+markers', name='MySQL', marker=dict(symbol="x", size=10), line=dict(color='black')))
        fig.add_trace(go.Scatter(x=Cassandra_uj, y=Cassandra_ul, mode='lines+markers', name='Cassandra', marker=dict(symbol="square", size=10), line=dict(color='gray')))
        fig.add_trace(go.Scatter(x=MongoDB_uj, y=MongoDB_ul, mode='lines+markers', name='MongoDB', marker=dict(symbol="circle", size=10), line=dict(color='lightgray')))

        fig.update_layout(
            title='Workload F: Update Latency',
            xaxis_title='Operations',
            yaxis_title='Update latency (μs)',
            legend_title='Database',
            template='simple_white'
        )
        return fig

    def plot_workload_g_write():
        MySQL_wl = [2983.49, 2552.61, 2883.77, 2911.42, 3056.96]
        MySQL_wj = [1000, 10000, 50000, 100000, 200000]
        MongoDB_wl = [351.12, 125.35, 101.55, 78.04, 96.96]
        MongoDB_wj = [1000, 10000, 50000, 100000, 200000]
        Cassandra_wl = [706.40, 348.43, 239.30, 265.56, 345.98]
        Cassandra_wj = [1000, 10000, 50000, 100000, 200000]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=MySQL_wj, y=MySQL_wl, mode='lines+markers', name='MySQL', marker=dict(symbol="x", size=10), line=dict(color='black')))
        fig.add_trace(go.Scatter(x=Cassandra_wj, y=Cassandra_wl, mode='lines+markers', name='Cassandra', marker=dict(symbol="square", size=10), line=dict(color='gray')))
        fig.add_trace(go.Scatter(x=MongoDB_wj, y=MongoDB_wl, mode='lines+markers', name='MongoDB', marker=dict(symbol="circle", size=10), line=dict(color='lightgray')))

        fig.update_layout(
            title='Workload G: Write Latency',
            xaxis_title='Operations',
            yaxis_title='Write latency (μs)',
            legend_title='Database',
            template='simple_white'
        )
        return fig

    def plot_workload_g_overall():
        MySQL_or = [3450, 25981, 144865, 292051, 612627]
        MySQL_ot = [289.86, 384.90, 345.15, 342.41, 326.46]
        MongoDB_or = [656, 1621, 5546, 8317, 20143]
        MongoDB_ot = [1524.39, 6169.03, 9015.51, 12023.57, 9929.01]
        Cassandra_or = [3419, 6278, 14864, 29534, 72361]
        Cassandra_ot = [292.48, 1595.86, 3363.83, 3385.93, 2763.92]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=MySQL_ot, y=MySQL_or, mode='lines+markers', name='MySQL', marker=dict(symbol="x", size=10), line=dict(color='black')))
        fig.add_trace(go.Scatter(x=Cassandra_ot, y=Cassandra_or, mode='lines+markers', name='Cassandra', marker=dict(symbol="square", size=10), line=dict(color='gray')))
        fig.add_trace(go.Scatter(x=MongoDB_ot, y=MongoDB_or, mode='lines+markers', name='MongoDB', marker=dict(symbol="circle", size=10), line=dict(color='lightgray')))

        fig.update_layout(
            title='Workload G: Overall',
            xaxis_title='Throughput (ops/sec)',
            yaxis_title='Runtime (ms)',
            legend_title='Database',
            template='simple_white'
        )
        return fig

    def plot_rows_load_inserts():
        MySQL_il = [2899.01, 2551.75, 3010.30, 2780.55, 3048.88] # latency
        MySQL_ij = [1000, 10000, 50000, 100000, 200000] # jobs
        MongoDB_il = [309.22, 115.94, 82.41, 87.61, 71.79]
        MongoDB_ij = [1000, 10000, 50000, 100000, 200000]
        Cassandra_il = [551.53, 382.41, 222.65, 246.60, 276.86]
        Cassandra_ij = [1000, 10000, 50000, 100000, 200000]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=MySQL_ij, y=MySQL_il, mode='lines+markers', name='MySQL', marker=dict(symbol="x", size=10), line=dict(color='black')))
        fig.add_trace(go.Scatter(x=Cassandra_ij, y=Cassandra_il, mode='lines+markers', name='Cassandra', marker=dict(symbol="square", size=10), line=dict(color='gray')))
        fig.add_trace(go.Scatter(x=MongoDB_ij, y=MongoDB_il, mode='lines+markers', name='MongoDB', marker=dict(symbol="circle", size=10), line=dict(color='lightgray')))

        fig.update_layout(
            title='Rows Load: Insert Latency',
            xaxis_title='Operations',
            yaxis_title='Insert latency (μs)',
            legend_title='Database',
            template='simple_white'
        )
        return fig

    def plot_rows_load_overall():
        MySQL_or = [3320, 26271, 151255, 278758, 610913] # Runtime
        MySQL_ot = [301.20, 380.65, 330.57, 358.73, 327.38] # throughput
        MongoDB_or = [715, 1524, 4524, 9288, 14981]
        MongoDB_ot = [1398.60, 6561.68, 11025.36, 10766.58, 13350.24]
        Cassandra_or = [3249, 6522, 13973, 27621, 58588]
        Cassandra_ot = [307.79, 1533.27, 3578.33, 3620.43, 3413.67]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=MySQL_ot, y=MySQL_or, mode='lines+markers', name='MySQL', marker=dict(symbol="x", size=10), line=dict(color='black')))
        fig.add_trace(go.Scatter(x=Cassandra_ot, y=Cassandra_or, mode='lines+markers', name='Cassandra', marker=dict(symbol="square", size=10), line=dict(color='gray')))
        fig.add_trace(go.Scatter(x=MongoDB_ot, y=MongoDB_or, mode='lines+markers', name='MongoDB', marker=dict(symbol="circle", size=10), line=dict(color='lightgray')))

        fig.update_layout(
            title='Rows Load: Overall',
            xaxis_title='Throughput (ops/sec)',
            yaxis_title='Runtime (ms)',
            legend_title='Database',
            template='simple_white'
        )
        return fig

    # Selectors for the workload visualization
    workloads = {
        "Workload A: Read Latency": plot_workload_a_read,
        "Workload A: Update Latency": plot_workload_a_update,
        "Workload B: Read Latency": plot_workload_b_read,
        "Workload B: Update Latency": plot_workload_b_update,
        "Workload C: Read Latency": plot_workload_c_read,
        "Workload C: Overall": plot_workload_c_overall,
        "Workload D: Read Latency": plot_workload_d_read,
        "Workload D: Insert Latency": plot_workload_d_insert,
        "Workload E: Scan Latency": plot_workload_e_scan,
        "Workload E: Insert Latency": plot_workload_e_insert,
        "Workload F: Read-modify-write Latency": plot_workload_f_read_modify_write,
        "Workload F: Update Latency": plot_workload_f_update,
        "Workload G: Write Latency": plot_workload_g_write,
        "Workload G: Overall": plot_workload_g_overall,
        "Rows Load: Insert Latency": plot_rows_load_inserts,
        "Rows Load: Overall": plot_rows_load_overall,
    }

    workload_selection = st.selectbox("Select Workload Visualization", list(workloads.keys()))
    st.plotly_chart(workloads[workload_selection](), use_container_width=True)

# Page 2: Twitter Sentiment Analysis Page
def twitter_sentiment_analysis_page():
    st.title("Twitter Sentiment Analysis")
    
    # Function to calculate hourly vader average with day
    def calculate_hourly_vader_average_with_day(df):
        df['vader_score'] = pd.to_numeric(df['vader_score'], errors='coerce')
        hourly_average = df.resample('h')['vader_score'].mean().reset_index()
        hourly_average['day_of_week'] = hourly_average['date'].dt.day_name()
        return hourly_average
    
    # Plot functions for Twitter Sentiment Analysis
    def plot_tweets_per_day(df):
        tweet_counts = df.groupby(df.index.date).size().reset_index(name='counts')
        tweet_counts['date'] = pd.to_datetime(tweet_counts['index'])
        fig = px.bar(tweet_counts, x='date', y='counts', title='Number of Tweets Per Day', labels={'date': 'Date', 'counts': 'Number of Tweets'}, color_discrete_sequence=['gray'])
        fig.update_layout(xaxis=dict(tickangle=0), title_x=0.5, template='simple_white')
        return fig

    def plot_daily_sentiment_average(df):
        sentiment_average = df.groupby(df.index.date)['vader_score'].mean().reset_index()
        sentiment_average['date'] = pd.to_datetime(sentiment_average['index'])
        fig = px.bar(sentiment_average, x='date', y='vader_score', title='Daily Sentiment Average', labels={'date': 'Date', 'vader_score': 'Vader Score'}, color_discrete_sequence=['gray'])
        fig.update_layout(xaxis=dict(tickangle=0), title_x=0.5, template='simple_white')
        return fig

    def plot_frequency_high_vader_scores(df):
        hourly_sentiment_average_with_day = calculate_hourly_vader_average_with_day(df)
        top_20_scores = hourly_sentiment_average_with_day.nlargest(20, 'vader_score').sort_values(by='date', ascending=True)
        day_counts = top_20_scores['day_of_week'].value_counts().reset_index()
        day_counts.columns = ['day_of_week', 'count']
        fig = px.bar(day_counts, x='day_of_week', y='count', title='Frequency of Days with Hourly Highest Vader Scores', labels={'day_of_week': 'Day of the Week', 'count': 'Frequency'}, color_discrete_sequence=['gray'])
        fig.update_layout(xaxis=dict(tickangle=0), title_x=0.5, template='simple_white')
        return fig

    def plot_frequency_low_vader_scores(df):
        hourly_sentiment_average_with_day = calculate_hourly_vader_average_with_day(df)
        lowest_20_scores = hourly_sentiment_average_with_day.nsmallest(20, 'vader_score').sort_values(by='date', ascending=True)
        day_counts_low = lowest_20_scores['day_of_week'].value_counts().reset_index()
        day_counts_low.columns = ['day_of_week', 'count']
        fig = px.bar(day_counts_low, x='day_of_week', y='count', title='Frequency of Days with Hourly Lowest Vader Scores', labels={'day_of_week': 'Day of the Week', 'count': 'Frequency'}, color_discrete_sequence=['gray'])
        fig.update_layout(xaxis=dict(tickangle=0), title_x=0.5, template='simple_white')
        return fig

    def plot_key_events(df):
        sentiment_average = df.groupby(df.index.date)['vader_score'].mean().reset_index()
        sentiment_average['date'] = pd.to_datetime(sentiment_average['index'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sentiment_average['date'], y=sentiment_average['vader_score'], mode='lines', line=dict(color='gray', width=2)))
        events = {
            'Positive 17/04/2009': ('2009-04-17\n Friday', '#Asot400\n #followfriday\n #readathon'),
            'Positive 10/05/2009': ('2009-05-10\n Sunday', '#SanctuarySunday\n #mothersday\n #140Sunday'),
            'Positive 17/05/2009': ('2009-05-17\n Sunday', '#eurovision\n @DavidArchie\n @mileycyrus'),
            'Positive 23/05/2009': ('2009-05-23\n Friday', '#followfriday\n #FF\n @ddlovato'),
            'Positive 31/05/2009': ('2009-05-31\n Sunday', '#andyhurleyday\n #MTVmovieawards\n @mileycyrus'),
            'Positive 05/06/2009': ('2009-06-05\n Friday', '@ddlovato'),
            'Negative 15/06/2009': ('2009-06-18\n Monday', '#iranelection')
        }
        for event, (date, label) in events.items():
            date_time = pd.to_datetime(date)
            fig.add_shape(type="line", x0=date_time, x1=date_time, y0=sentiment_average['vader_score'].min(), y1=sentiment_average['vader_score'].max(), line=dict(color='red', dash='dash'))
            fig.add_annotation(x=date_time, y=sentiment_average['vader_score'].max() * 0.3, text=label, showarrow=False, textangle=60, font=dict(color='darkred'))
        fig.update_layout(title='Key Events | Daily Average Vader Score', xaxis_title='Date', yaxis_title='VADER Sentiment Score', xaxis=dict(tickangle=0), template='simple_white')
        return fig

    sentiment_options = {
        "Number of Tweets Per Day": plot_tweets_per_day,
        "Daily Sentiment Average": plot_daily_sentiment_average,
        "Frequency of Days with Hourly Highest Vader Scores": plot_frequency_high_vader_scores,
        "Frequency of Days with Hourly Lowest Vader Scores": plot_frequency_low_vader_scores,
        "Key Events": plot_key_events,
    }

    sentiment_selection = st.selectbox("Select Sentiment Analysis Visualization", list(sentiment_options.keys()))
    st.plotly_chart(sentiment_options[sentiment_selection](df_twitter), use_container_width=True)

# Page 3: ForecasterAutoreg Time Series
def forecaster_autoreg_page():
    st.title("ForecasterAutoreg Time Series")
    options = ["Initial Model", "Hyperparameter Tuned Model", "Summary"]
    selection = st.selectbox("Select Option", options)

    # Initializing the forecaster
    forecaster = ForecasterAutoreg(
        regressor=RandomForestRegressor(random_state=123),
        lags=24  # Use 24 lags since data is hourly
    )

    # Splitting data - leaving out the last 7 days for testing
    train = hsa3[:-24*7]
    test = hsa3[-24*7:]

    # Fitting the forecaster with the training data
    forecaster.fit(y=train['vader_score'])

    if selection == "Initial Model":
        # List of prediction steps and titles
        steps_list = [24, 72, 168]
        titles = ['One Day Sentiment Prediction', 'Three Day Sentiment Prediction', 'Seven Day Sentiment Prediction']

        # Dictionary to hold MSE values
        mse_values = {}

        # Plotting the predictions for different steps
        for steps, title in zip(steps_list, titles):
            # Predict future values beyond the available data
            last_timestamp = hsa3.index[-1]
            prediction_start_date = last_timestamp + pd.Timedelta(hours=1)
            predictions = forecaster.predict(steps=steps, last_window=hsa3['vader_score'].tail(forecaster.window_size))
            predictions.index = pd.date_range(start=prediction_start_date, periods=len(predictions), freq='h')

            # Calculate MSE
            mse = mean_squared_error(test['vader_score'].iloc[:steps], predictions)
            mse_values[title] = mse

            # Plotting
            plt.figure(figsize=(15, 7))
            plt.plot(hsa3['vader_score'], label='Train data')
            plt.plot(test.index, test['vader_score'], label='Test data', color='orange')
            plt.plot(predictions, label='Future Predictions', color='red')
            plt.title(title)
            plt.xlabel('Date')
            plt.ylabel('Sentiment')
            plt.legend()
            st.pyplot(plt)

    elif selection == "Hyperparameter Tuned Model":
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20],
            'min_samples_split': [2, 5, 10]
        }

        best_mse = {24: np.inf, 72: np.inf, 168: np.inf}
        best_params = {24: {}, 72: {}, 168: {}}
        best_predictions = {24: None, 72: None, 168: None}

        for params in ParameterGrid(param_grid):
            forecaster.regressor = RandomForestRegressor(random_state=123, **params)
            forecaster.fit(y=train['vader_score'])
            for steps in [24, 72, 168]:
                predictions = forecaster.predict(steps=steps)
                mse = mean_squared_error(test['vader_score'].iloc[:steps], predictions)
                if mse < best_mse[steps]:
                    best_mse[steps] = mse
                    best_params[steps] = params
                    best_predictions[steps] = predictions

        # Plotting the best predictions for each period
        periods = [24, 72, 168]
        titles = ['Best One Day Sentiment Prediction', 'Best Three Day Sentiment Prediction', 'Best Seven Day Sentiment Prediction']

        for period, title in zip(periods, titles):
            prediction_start_date = test.index[-1] + pd.Timedelta(hours=1)
            best_predictions[period].index = pd.date_range(start=prediction_start_date, periods=len(best_predictions[period]), freq='h')

            plt.figure(figsize=(15, 7))
            plt.plot(hsa3['vader_score'], label='Train data')
            plt.plot(test.index, test['vader_score'], label='Test data', color='orange')
            plt.plot(best_predictions[period], label='Best Predictions', color='green')
            plt.title(title)
            plt.xlabel('Date')
            plt.ylabel('Sentiment')
            plt.legend()
            st.pyplot(plt)

    elif selection == "Summary":
        # Calculate initial model MSE
        mse_24 = mean_squared_error(test['vader_score'].iloc[:24], forecaster.predict(steps=24))
        mse_72 = mean_squared_error(test['vader_score'].iloc[:72], forecaster.predict(steps=72))
        mse_168 = mean_squared_error(test['vader_score'].iloc[:168], forecaster.predict(steps=168))

        # Hyperparameter tuning MSE
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20],
            'min_samples_split': [2, 5, 10]
        }

        best_mse = {24: np.inf, 72: np.inf, 168: np.inf}
        best_params = {24: {}, 72: {}, 168: {}}

        for params in ParameterGrid(param_grid):
            forecaster.regressor = RandomForestRegressor(random_state=123, **params)
            forecaster.fit(y=train['vader_score'])
            for steps in [24, 72, 168]:
                predictions = forecaster.predict(steps=steps)
                mse = mean_squared_error(test['vader_score'].iloc[:steps], predictions)
                if mse < best_mse[steps]:
                    best_mse[steps] = mse
                    best_params[steps] = params

        # Summary
        results = {
            "Forecast Period": [
                "1 day",
                "1 day (Best)",
                "3 days",
                "3 days (Best)",
                "7 days",
                "7 days (Best)"
            ],
            "MSE": [
                mse_24,
                best_mse[24],
                mse_72,
                best_mse[72],
                mse_168,
                best_mse[168]
            ]
        }

        # Create the DataFrame
        summary = pd.DataFrame(results)
        st.write("Summary of Mean Squared Errors (MSE):")
        st.dataframe(summary)

# Page 4: LSTM Time Series
def lstm_time_series_page():
    st.title("LSTM Time Series")

    # Splitting data
    train = hsa3['vader_score'][:-24*22]
    test = hsa3['vader_score'][-24*22:]

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    test_scaled = scaler.transform(test.values.reshape(-1, 1))

    # Create sequences
    time_step = 24  # Using 24 hours as the time step
    X_train, y_train = create_sequences(train_scaled, time_step)
    X_test, y_test = create_sequences(test_scaled, time_step)

    # Reshape input to be [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build the LSTM model
    model = Sequential()
    model.add(Input(shape=(time_step, 1)))  # Use Input layer as the first layer
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=1, epochs=10)

    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    # Calculate RMSE for test predictions
    rmse_test = np.sqrt(mean_squared_error(test[time_step:], predictions))

    # Adjust test index for plotting
    test_index = test.index[time_step:]

    plot_lstm_results(train, test, test_index, predictions, rmse_test)
    plot_lstm_forecasts(train, test, test_index, predictions, model, test_scaled, time_step, scaler)

# Navigation
st.sidebar.title("Menu")
page = st.sidebar.radio("Go to section:", ["YCSB Workloads", "Twitter Sentiment Analysis", "Time Series ForecasterAutoreg", "Time Series LSTM"])

# Load the appropriate page
if page == "YCSB Workloads":
    ycsb_workloads_page()
elif page == "Twitter Sentiment Analysis":
    twitter_sentiment_analysis_page()
elif page == "Time Series ForecasterAutoreg":
    forecaster_autoreg_page()
elif page == "Time Series LSTM":
    lstm_time_series_page()

# Additional tags
st.markdown("<p style='text-align: center;'>Developed with ❤️ at CCT College Dublin.</p>", unsafe_allow_html=True)