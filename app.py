import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objs as go
import plotly.express as px

# Judul Aplikasi
st.title('Analisis Triple Exponential Smoothing')
st.subheader("Zulaikha Anissa Azfa 24050121140114")

# Input di Sidebar
st.sidebar.header('Pengaturan Input')

# Upload File CSV
uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    # Membaca data
    data = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)

    # Memilih kolom untuk analisis
    column = st.sidebar.selectbox("Pilih kolom untuk analisis", data.columns)

    # Set ulang indeks data
    data.reset_index(drop=True, inplace=True)

    # Menentukan proporsi train-test split
    train_size = st.sidebar.slider(
        "Pilih proporsi data untuk training (dalam %)", 50, 95, 80) / 100.0

    # Memilih parameter untuk Triple Exponential Smoothing
    seasonal_periods = st.sidebar.slider("Pilih periode musiman", 1, 12, 12)
    trend = st.sidebar.selectbox(
        "Pilih jenis trend", ["additive", "multiplicative"])
    seasonal = st.sidebar.selectbox(
        "Pilih jenis musim", ["additive", "multiplicative"])

    forecast_period_unseen = st.sidebar.slider(
        "Pilih periode prediksi", 1, 36, 6)

    # Membagi data menjadi train dan test set
    train_data_len = int(len(data) * train_size)
    train_data = data.iloc[:train_data_len]
    test_data = data.iloc[train_data_len:]

    # Menerapkan metode Triple Exponential Smoothing pada data train
    model = ExponentialSmoothing(
        train_data[column],
        seasonal_periods=seasonal_periods,
        trend=trend,
        seasonal=seasonal
    ).fit()

    # Membuat prediksi pada data test
    forecast_period = len(test_data)
    forecast = model.forecast(forecast_period)

    # Membuat prediksi pada data unseen
    model_unseen = ExponentialSmoothing(
        data[column],
        seasonal_periods=seasonal_periods,
        trend=trend,
        seasonal=seasonal
    ).fit()
    forecast_unseen = model_unseen.forecast(forecast_period_unseen)

    # Membagi output menjadi beberapa tab
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Preview Data", "EDA", "Pemodelan dan Evaluasi", 'Hasil Prediksi'])

    with tab1:
        st.header("Preview Data")

        # Plot time series data asli
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=data.index, y=data[column], mode='lines', name='Data Asli'))
        fig1.update_layout(title='Grafik Time Series Data Asli',
                           xaxis_title='Tanggal',
                           yaxis_title='Nilai')
        st.plotly_chart(fig1)

        st.write("5 Data Pertama:")
        st.write(data.head())

    with tab2:
        st.header("EDA (Exploratory Data Analysis)")

        # Grafik dekomposisi
        decomposed = seasonal_decompose(
            data[column], model=trend, period=seasonal_periods)

        st.subheader("Trend")
        fig2_trend = go.Figure()
        fig2_trend.add_trace(go.Scatter(
            x=decomposed.trend.index, y=decomposed.trend, mode='lines', name='Trend'))
        fig2_trend.update_layout(title='Dekomposisi Trend',
                                 xaxis_title='Tanggal',
                                 yaxis_title='Nilai')
        st.plotly_chart(fig2_trend)

        st.subheader("Seasonal")
        fig2_seasonal = go.Figure()
        fig2_seasonal.add_trace(go.Scatter(
            x=decomposed.seasonal.index, y=decomposed.seasonal, mode='lines', name='Seasonal'))
        fig2_seasonal.update_layout(title='Dekomposisi Seasonal',
                                    xaxis_title='Tanggal',
                                    yaxis_title='Nilai')
        st.plotly_chart(fig2_seasonal)

        st.subheader("Residual")
        fig2_residual = go.Figure()
        fig2_residual.add_trace(go.Scatter(
            x=decomposed.resid.index, y=decomposed.resid, mode='lines', name='Residual'))
        fig2_residual.update_layout(title='Dekomposisi Residual',
                                    xaxis_title='Tanggal',
                                    yaxis_title='Nilai')
        st.plotly_chart(fig2_residual)

        # Grafik distribusi data
        st.subheader("Distribusi Data")
        fig3 = px.histogram(data[column], nbins=50, title='Distribusi Data')
        st.plotly_chart(fig3)

    with tab3:
        st.header("Pemodelan")

        # Plot data asli dan hasil prediksi menggunakan Plotly
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=train_data.index, y=train_data[column], mode='lines', name='Data Train', line=dict(color='blue')))
        fig4.add_trace(go.Scatter(
            x=test_data.index, y=test_data[column], mode='lines', name='Data Test', line=dict(color='orange')))
        fig4.add_trace(go.Scatter(x=forecast.index, y=forecast,
                       mode='lines', name='Hasil Prediksi', line=dict(color='green')))
        fig4.update_layout(title='Hasil Analisis Triple Exponential Smoothing',
                           xaxis_title='Tanggal',
                           yaxis_title='Nilai',
                           hovermode='x unified')
        st.plotly_chart(fig4)

        st.header("Evaluasi")

        # Menghitung metric evaluasi
        residuals = test_data[column] - forecast
        mse = mean_squared_error(test_data[column], forecast)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_data[column], forecast)
        mape = np.mean(np.abs(residuals / test_data[column])) * 100
        smape = 2.0 * np.mean(np.abs(residuals) /
                              (np.abs(test_data[column]) + np.abs(forecast))) * 100

        st.write(f"""
            **MSE:** {mse:.2f} &nbsp; &nbsp; 
            **RMSE:** {rmse:.2f} &nbsp; &nbsp; 
            **MAE:** {mae:.2f} &nbsp; &nbsp; 
            **MAPE:** {mape:.2f}% &nbsp; &nbsp; 
            **SMAPE:** {smape:.2f}%
        """)

        # Plot persebaran residual
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=residuals.index, y=residuals,
                       mode='markers', name='Residual'))
        fig5.update_layout(title='Persebaran Residual',
                           xaxis_title='Tanggal',
                           yaxis_title='Residual')
        st.plotly_chart(fig5)

    with tab4:
        st.header("Hasil Prediksi")

        # Plot data asli dan hasil prediksi menggunakan Plotly
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(
            x=data.index, y=data[column], mode='lines', name='Data Asli', line=dict(color='blue')))
        fig6.add_trace(go.Scatter(
            x=forecast_unseen.index, y=forecast_unseen, mode='lines', name='Hasil Prediksi', line=dict(color='green')))
        fig6.update_layout(title='Hasil Prediksi Triple Exponential Smoothing',
                           xaxis_title='Tanggal',
                           yaxis_title='Nilai',
                           hovermode='x unified')
        st.plotly_chart(fig6)

        csv = forecast_unseen.to_csv().encode('utf-8')
        st.download_button(
            label="Download Hasil Prediksi",
            data=csv,
            file_name='hasil_prediksi.csv',
            mime='text/csv',
        )
else:
    st.info("Upload file CSV untuk memulai analisis.")
