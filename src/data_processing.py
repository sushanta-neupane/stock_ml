import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Functions for data processing and model creation
def prepare_data(df, time_step=100, split_ratio=0.65):
    df2 = df['close']
    scaler = MinMaxScaler()
    df2_scaled = scaler.fit_transform(np.array(df2).reshape(-1, 1))
    
    train_size = int(len(df2_scaled) * split_ratio)
    train_data, test_data = df2_scaled[0:train_size], df2_scaled[train_size:]
    X_train, Y_train = create_dataset(train_data, time_step)
    X_test, Y_test = create_dataset(test_data, time_step)
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    return X_train, Y_train, X_test, Y_test, scaler

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)
