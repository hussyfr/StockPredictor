#Load Netflix Stock Data
import pandas as pd
stock_data = pd.read_csv('./NFLX.csv',index_col='Date')
stock data.head)

#Necessary imports for data visualization
import matplotlib.dates as mates
import matplotlib.pyplot as plt
import datetime as dt

#We will use the Open, High, and Low columns to predict the 
#Closing value of the Netflix stock for the next day
plt.figure(figsize=(15,10))
plt.gca().xaxis.set major formatter (mates.DateFormatter ('%Y-%m-%d') )
plt.ca().xaxis.set major locator(mdates. DayLocator(interval=60))
x dates = [dt.datetime.strptime(d, '%Y-%m-%d*) .date() for d in stock_data. index.values]

plt.plot (×_dates, stock_data[ 'High'], label='High')
plt.plot (×_dates, stock_data[ 'Low'], label='Low')
plt.xlabel ('Time Scale'.
plt.ylabel('Scaled USD'
plt. legend ()
plt.gcf () . autofmt_xdate ()
plt. show()

#More necessary imports for price prediction
import numpy as np
from tensorflow. keras.models import Sequential
from tensorflow.keras. layers import Dense
from tensorflow. keras.layers import LSTM
from tensorflow.keras. layers import Dropout
from tensonflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model selection import train test split
from sklearn.model selection import TimeSeriesSplit
from sklearn.metrics import mean squared_ error

#Preprocessing
target y = stock data 'close'
X feat = stock data.iloc[:,0:3]
 
#Rescale Values
Sc = StandardScaler ()
X ft = sc.fit transform(X feat.values)
X_ft = pd. DataFrame (columns=X_feat. columns,
data=X_ft, index=X_feat.index)

#Splitting data
def lstm_split (data, n_steps):
    X, y = [], []
    for i in range(len (data) -n_steps+1):
        X. append (data[i:1 + n_steps, : -1])
        y.append(data[i + n_steps-1, -1])
    return np.array(X), np.array (y)

#Train and test sets
X1, y1 = lstm_split(stock_data_ft. values, n_steps=2)
train split=0.8
split_idx = int{np.ceil(len(X1)*train_split))
date_index = stock_data_ft.index
X_train, X_test = X1[:split_idx], X1[split_idx:]
y_train, y_test = y1[:split_idx], y1[split_idx: ]
×_ train_date, X_test_date = date_index[:split_idx], date_index[split_idx:]
print (XI. shape, X_train.shape, ×_test.shape, _test.shape)

#Building LSTM model
lstm = Sequential ()
lstm.add(LSTM(50, input_shape=(X_train.shape[1], ×_ train.shape[2]),
activation='relu', return sequences=True))
lstm. add(LSTM(50, activation='relu'))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')
lstm.summary ()

history-lstm. fit(X_train, y_train,
                  epochs=100, batch_ size=4,
                  verbose=2, shuffle=False)

#Performance evaluation
rmse = mean_squared_error (_test, y_pred, squared=False)
mape = mean_absolute_percentage_error (y_test, y_pred)
print ("RSME:", rmse)
print("MAPE:", mape)


