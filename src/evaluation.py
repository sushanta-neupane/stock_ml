import numpy as np
import math
from sklearn.metrics import mean_squared_error

def calculate_accuracy(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if len(y_true) > 0 else float('inf')
    return rmse, mape
