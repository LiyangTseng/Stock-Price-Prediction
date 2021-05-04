import tensorflow.keras as keras
import os
import numpy as np
import pandas as pd
from datetime import date

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
    return data_df

def main():
    month, day = date.today().month, date.today().day
    ID = '0611097'
    submission_filename = '{:02d}{:02d}_{}.txt'.format(month, day, ID)
    
    num_pastRecord, num_predictRecord = 30, 1
    model_path = 'model'
    model = keras.models.load_model(model_path)
    data_dir = 'Company_Data'
    company_list = ['INTC', 'AMD', 'CSCO', 'AAPL', 'MU', 'NVDA', 'QCOM', 'AMZN', 'NFLX', 'FB',
                 'GOOG', 'BABA', 'EBAY', 'IBM', 'XLNX', 'TXN', 'NOK', 'TSLA', 'MSFT', 'SNPS']
    prediction_list = []

    print('========== predict adjusted closing price via inference.py ==========')
    for company in company_list:
        stock_data_path = os.path.join(data_dir, company+'.csv')
        stock_data = read_data(stock_data_path)
        
        features = np.array(stock_data.iloc[len(stock_data)-num_pastRecord:])
        features = features[np.newaxis, :, :]
        prediction = model.predict(features)
        prediction_list.append(np.argmax(prediction))

    with open(submission_filename, 'w') as f:
        for ele in prediction_list:
            f.write('%s\n' % ele) 
    print('output predictions to {file}!'.format(file=submission_filename))

if __name__ == '__main__':
    main()

