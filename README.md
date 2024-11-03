# Stock Price Prediction App 📈

A Streamlit-based web application for predicting stock prices using machine learning. This application allows users to upload stock data, train models, and visualize predictions.

## Features ✨

- Upload and preview stock data in CSV format
- Train new LSTM models for prediction
- Use pre-trained models or upload custom models
- Real-time prediction visualization
- Performance metrics (RMSE and MAPE)
- Interactive web interface

## Prerequisites 🛠️

Before running the application, ensure you have Python 3.7+ installed on your system.

## Installation 📦

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-prediction-app.git
cd stock-prediction-app
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirement.txt
```

### Requirements

Create a `requirements.txt` file with the following dependencies:
```
streamlit>=1.0.0
pandas>=1.3.0
numpy>=1.19.5
matplotlib>=3.4.3
scikit-learn>=0.24.2
tensorflow>=2.7.0
```

## Project Structure 📁

```
stock-prediction-app/
├── app.py
├── requirements.txt
├── README.md
├── models/
└── src/
    ├── __init__.py
    ├── data_processing.py
    ├── evaluation.py
    ├── model.py
    └── visualization.py
```

## Usage 💡

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Access the application through your web browser (typically at `http://localhost:8501`)

3. Follow the steps in the application:
   - **Step 1:** Upload your stock data CSV file
     - The CSV file should contain at least a 'close' price column
     - Data should be in chronological order
   
   - **Step 2:** Select or train a model
     - Choose a pre-existing model from the 'models' folder
     - Upload your own custom model (.h5 format)
     - Or train a new model using the "Train New Model" button
   
   - **Step 3:** View predictions
     - Examine the prediction results visualization
     - Review performance metrics (RMSE and MAPE)

## Data Format 📊

Your input CSV file should contain the following columns:
- `close`: Required - The closing price of the stock
- Additional columns may be present but aren't used in the current implementation

Example CSV format:
```
date,close,volume
2023-01-01,150.25,1000000
2023-01-02,151.50,1200000
...
```

## Model Training Parameters 🧠

The default LSTM model uses the following configuration:
- Epochs: 50
- Batch Size: 64
- Sequence Length: Determined by input data
- Architecture: LSTM layers with dropout

## Performance Metrics 📊

The app provides two key performance metrics:
- **RMSE (Root Mean Square Error)**: Measures the standard deviation of prediction errors
- **MAPE (Mean Absolute Percentage Error)**: Measures prediction accuracy as a percentage

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

- TensorFlow team for the deep learning framework
- Streamlit team for the wonderful web app framework

## Support 💬

For support, please open an issue in the GitHub repository or contact [your-email@example.com]
