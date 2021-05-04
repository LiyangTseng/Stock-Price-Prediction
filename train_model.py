import os
import pandas as pd
import numpy as np
import tensorflow as tf
import sys
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
company_list = ['INTC', 'AMD', 'CSCO', 'AAPL', 'MU', 'NVDA', 'QCOM', 'AMZN', 'NFLX', 'FB',
                 'GOOG', 'BABA', 'EBAY', 'IBM', 'XLNX', 'TXN', 'NOK', 'TSLA', 'MSFT', 'SNPS']

def read_data(stock_data_path):
    data_df = pd.read_csv(stock_data_path).sort_values('Unnamed: 0', ascending=True)
    data_df = data_df.drop(['Unnamed: 0', '7. dividend amount','8. split coefficient'], axis=1)
    data_df['status'] = -1
    for index, row in data_df.iterrows():
        if index != len(data_df)-1: #discard first row
            rate_of_change = (data_df.loc[index, '5. adjusted close'] - data_df.loc[index+1, '5. adjusted close']) / (
                data_df.loc[index+1, '5. adjusted close'])
            if rate_of_change > 0.015: # rise
                data_df.loc[index, 'status'] = 0
            elif rate_of_change < -0.015: # fall
                data_df.loc[index, 'status'] = 2
            else:                       # stable
                data_df.loc[index, 'status'] = 1
    data_df = data_df.drop([len(data_df)-1])
    return data_df.to_numpy()

def get_feature_label_pairs(train, pastDay, futureDay):
    X_train, Y_train = [], []
    for i in range(train.shape[0]-futureDay-pastDay):
        # X_train.append(np.array(train.iloc[i:i+pastDay]))
        # Y_train.append(np.array(train.iloc[i+pastDay:i+pastDay+futureDay]['status']))
        X_train.append(train[i:i+pastDay, :])
        Y_train.append(train[i+pastDay:i+pastDay+futureDay, -1])
    return np.array(X_train), np.array(Y_train)

def split_data(X,Y,rate):
    # X_train = np.array(X[int(len(X)*rate):])
    # Y_train = np.array(Y[int(len(Y)*rate):])
    # X_val =   np.array(X[:int(len(X)*rate)])
    # Y_val =   np.array(Y[:int(len(Y)*rate)])
    X_train = X[int(len(X)*rate):, :, :]
    Y_train = Y[int(len(Y)*rate):, :]
    X_val =   X[:int(len(X)*rate), :, :]
    Y_val =   Y[:int(len(Y)*rate):, :]
    return X_train, Y_train, X_val, Y_val

class LSTM_MODEL(keras.Model):
    def __init__(self, shape):
        super(LSTM_MODEL, self).__init__()
        self.BatchNorm1 = keras.layers.BatchNormalization()
        self.Dense1 = keras.layers.Dense(32, input_shape=(shape[1], shape[2]) )
        self.Activation1 = keras.layers.Activation('relu')
        self.BatchNorm2 = keras.layers.BatchNormalization()
        self.LSTM1 = keras.layers.LSTM(90, return_sequences=True) 
                                        # kernel_regularizer=l2(1e-4),  recurrent_regularizer=l2(1e-4))
        self.BatchNorm3 = keras.layers.BatchNormalization()
        self.LSTM2 = keras.layers.LSTM(150, return_sequences=True) 
                                        # kernel_regularizer=l2(1e-4),  recurrent_regularizer=l2(1e-4))
        self.BatchNorm4 = keras.layers.BatchNormalization()
        self.LSTM3 = keras.layers.LSTM(120, return_sequences=False) 
                                        # kernel_regularizer=l2(1e-4),  recurrent_regularizer=l2(1e-4))
        self.BatchNorm5 = keras.layers.BatchNormalization()
        self.Dropout = keras.layers.Dropout(0.15)
        self.Dense2 = keras.layers.Dense(64)
        self.BatchNorm6 = keras.layers.BatchNormalization()
        self.Activation2 = keras.layers.Activation('relu')
        self.Dense3 = keras.layers.Dense(3)
        self.Activation3 = keras.layers.Activation('softmax')
        
    def call(self, inputs):
        x = self.BatchNorm1(inputs)
        x = self.Dense1(x)
        x = self.Activation1(x)
        x = self.BatchNorm2(x)
        x = self.LSTM1(x)
        x = self.BatchNorm3(x)
        x = self.LSTM2(x)
        x = self.BatchNorm4(x)
        x = self.LSTM3(x)
        x = self.BatchNorm5(x)
        x = self.Dropout(x)
        x = self.Dense2(x)
        x = self.BatchNorm6(x)
        x = self.Activation2(x)
        x = self.Dense3(x)
        y = self.Activation3(x)        
        return y

def main():
    num_pastRecord, num_predictRecord = 30, 1
    data_dir = 'Company_Data'
    # features, labels = [], []
    features, labels = np.empty((0, num_pastRecord, 24)), np.empty((0, 1))
    for stock_data_subpath in os.listdir(data_dir):
        stock_data_path = os.path.join(data_dir, stock_data_subpath)
        stock_data = read_data(stock_data_path)
        stock_features, stock_labels = get_feature_label_pairs(stock_data, num_pastRecord, num_predictRecord)
        # features.extend(stock_features)
        # labels.extend(stock_labels)
        features = np.append(features, stock_features, axis=0)
        labels = np.append(labels, stock_labels, axis=0)

    labels = to_categorical(labels, num_classes=3) 
    train_features, train_labels, val_features, val_labels = split_data(features, labels, 0.1)


    model = LSTM_MODEL(train_features.shape)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'], run_eagerly=True)
    # model.summary()
    
    best_weight_path = 'best_weights.h5'
    callback = EarlyStopping(monitor="val_loss", patience=10, verbose=1, mode="auto")
    mcp_save = ModelCheckpoint('best_weights.h5', save_best_only=True, monitor='val_loss', mode='min')
    model.fit(train_features, train_labels, epochs=100, batch_size=128, validation_data=(val_features, val_labels), callbacks=[callback, mcp_save])
    model.load_weights(best_weight_path)
    model.save('model', save_format='tf')

if __name__ == '__main__':
    main()