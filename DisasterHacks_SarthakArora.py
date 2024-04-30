import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping

df_cod = pd.read_csv('/content/drive/MyDrive/Hackathon/wfp_food_prices_cod.csv')
df_eth = pd.read_csv('/content/drive/MyDrive/Hackathon/wfp_food_prices_eth.csv')
df_ken = pd.read_csv('/content/drive/MyDrive/Hackathon/wfp_food_prices_ken.csv')
df_som = pd.read_csv('/content/drive/MyDrive/Hackathon/wfp_food_prices_som.csv')
df_sdn = pd.read_csv('/content/drive/MyDrive/Hackathon/wfp_food_prices_sdn.csv')
df_ssd = pd.read_csv('/content/drive/MyDrive/Hackathon/wfp_food_prices_ssd.csv')

def preprocess_data(df, commodity):
    df = df.drop(0)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    if commodity == 'overall':
        df_commodity = df.groupby('date')['price'].sum().reset_index()
    else:
        df_commodity = df[df['commodity'].str.lower().str.contains(commodity.lower())]
    df_commodity = df_commodity[['date', 'price']]
    df_commodity = df_commodity.set_index('date')
    df_commodity = df_commodity.resample('D').mean()
    df_commodity = df_commodity.fillna(method='ffill')
    df_commodity = df_commodity.dropna()
    df_commodity = df_commodity.reset_index()
    df_commodity = df_commodity.set_index('date')
    df_commodity = df_commodity['price']
    return df_commodity

def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

def train_model(data, time_step=100):
    data = data.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    training_size = int(len(data) * 0.8)
    test_size = len(data) - training_size
    train_data, test_data = data[0:training_size, :], data[training_size:len(data), :]
    X_train, Y_train = create_dataset(train_data, time_step)
    X_test, Y_test = create_dataset(test_data, time_step)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='loss', patience=2)
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=64, verbose=1, callbacks=[early_stopping])
    return model, scaler, time_step, X_train, X_test, data #test_data

def predict_prices_for_countries(df, commodity, country, days=30):
      df_op = preprocess_data(df, commodity)
      model_op, scaler_op, time_step_op, X_train, X_test, data = train_model(df_op, 5)

      train_predict = model_op.predict(X_train)
      test_predict = model_op.predict(X_test)
      train_predict = scaler_op.inverse_transform(train_predict)
      test_predict = scaler_op.inverse_transform(test_predict)
      look_back = time_step_op
      trainPredictPlot = np.empty_like(data)
      trainPredictPlot[:] = np.nan
      trainPredictPlot[look_back:len(train_predict) + look_back] = train_predict
      testPredictPlot = np.empty_like(data)
      testPredictPlot[:] = np.nan
      testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(data) - 1] = test_predict
      plt.figure(figsize=(10,6))
      plt.plot(scaler_op.inverse_transform(data))
      plt.plot(trainPredictPlot)
      plt.plot(testPredictPlot)
      if commodity == 'overall':
        plt.title('Price for all Commodities in '+country)
      else:
        plt.title(commodity+' price in '+ country)
      plt.show()

predict_prices_for_countries(df_cod, 'overall', 'Congo', 30)