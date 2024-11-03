import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def plot_results(df, train_predict, test_predict, scaler):
    trainPredictPlot = np.empty_like(df)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[100:len(train_predict) + 100, :] = train_predict

    testPredictPlot = np.empty_like(df)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict) + 200:len(df), :] = test_predict

    plt.figure(figsize=(10, 5))
    plt.plot(scaler.inverse_transform(df), label='Actual Prices', color='blue')
    plt.plot(trainPredictPlot, label='Train Predictions', color='orange')
    plt.plot(testPredictPlot, label='Test Predictions', color='green')
    plt.title('LSTM Predictions vs Actual Prices')
    plt.xlabel('Days')
    plt.ylabel('Close Price')
    plt.legend()
    st.pyplot(plt)
