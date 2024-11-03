import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os

from src.data_processing import prepare_data
from src.evaluation import calculate_accuracy
from src.model import build_model
from src.visualization import plot_results


# Streamlit UI Design
st.title(" Stock Price Prediction App")

# Step 1: Upload the CSV file
st.markdown("### Step 1: Upload Your Stock Data CSV File")
uploaded_file = st.file_uploader("Choose a CSV file with stock data", type="csv")

if uploaded_file:
    # Display dataset preview
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.write("### Dataset Preview:", df.head())

    # Prepare data
    X_train, Y_train, X_test, Y_test, scaler = prepare_data(df)
    
    # Model selection section
    st.markdown("---")
    st.markdown("### Step 2: Model Selection")
    
    # Check for models in "models" folder
    models_folder = 'models'
    os.makedirs(models_folder, exist_ok=True)  # Ensure models folder exists
    available_models = [f for f in os.listdir(models_folder) if f.endswith('.h5')]
    
    if available_models:
        selected_model = st.selectbox("Select a model from the available models:", available_models)
        load_custom_model = st.checkbox("Or upload a custom model file")
    else:
        st.warning("⚠️ No models found in the 'models' folder.")
        load_custom_model = st.checkbox("Upload a custom model file")
    
    # Custom model uploader if checkbox is selected
    if load_custom_model:
        custom_model_file = st.file_uploader("Upload your own model file (.h5)", type="h5")
    else:
        custom_model_file = None
    
    # Option to train a new model
    if st.button("Train New Model"):
        with st.spinner("Training the model..."):
            model = build_model((X_train.shape[1], 1))
            model.fit(X_train, Y_train, epochs=50, batch_size=64, verbose=1)
            model_path = os.path.join(models_folder, 'stock_lstm_model.h5')
            model.save(model_path)
            st.success(f"🎉 Model trained and saved as {model_path}")

    # Load selected or uploaded model
    elif st.button("Load Model"):
        if custom_model_file:
            model = load_model(custom_model_file)
            st.success("📂 Custom model loaded successfully!")
        elif selected_model:
            model_path = os.path.join(models_folder, selected_model)
            model = load_model(model_path)
            st.success(f"📂 Model '{selected_model}' loaded successfully!")
        else:
            st.error("⚠️ No model selected or uploaded. Please choose a model to load.")

    # Prediction and plotting if model is loaded
    if 'model' in locals():
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        
        # Inverse scale the predictions
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        
        # Display accuracy metrics
        train_rmse, train_mape = calculate_accuracy(Y_train, train_predict)
        test_rmse, test_mape = calculate_accuracy(Y_test, test_predict)
        
        st.write(f"**Train RMSE:** {train_rmse:.2f}")
        st.write(f"**Train MAPE:** {train_mape:.2f}%")
        st.write(f"**Test RMSE:** {test_rmse:.2f}")
        st.write(f"**Test MAPE:** {test_mape:.2f}%")
        
        # Plot results
        st.markdown("---")
        st.markdown("### Step 3: View Predictions")
        plot_results(scaler.transform(np.array(df['close']).reshape(-1, 1)), train_predict, test_predict, scaler)
