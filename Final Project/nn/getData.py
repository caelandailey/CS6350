import numpy as np
import pandas as pd
from datetime import datetime
from datetime import date

class getData:

    def __init__(self):
        print('init get data')

    
    def get_data():

        # Setup
        date = 'date'
        timestamp = 'Timestamp'
        weighted_price = 'Weighted_Price'
        unit = 's'
        
        data = pd.read_csv('../data/bitcoin_data.csv')
        
        # Sort
        data[date] = pd.to_datetime(data[timestamp],unit=unit).dt.date
        Daily = data.groupby(date)[weighted_price].mean()

        # Get days
        days_start = (date(2017, 10, 15) - date(2016, 1, 1)).days + 1
        days_train = (date(2017, 10, 20) - date(2017, 8, 21)).days + 1
        days_end = (date(2017, 10, 20) - date(2017, 10, 15)).days + 1

        # Group
        daily_len = len(Daily)
        train= Daily[daily_len-days_start-days_end:daily_len-days_train]
        test= Daily[daily_len-days_train:]
        
        sub = [train, test]
        sub = pd.concat(working_data)
        sub = sub.reset_index()
        sub[date] = pd.to_datetime(sub[date])
        sub = sub.set_index(date)

        train = sub[:-60]
        test = sub[-60:]

        # Get set
        train_set = train.values
        len_training = len(train_set)
        train_set = np.reshape(train_set, len_training, 1))
        test_set = test.values
        test_set = np.reshape(test_set, len_training, 1))

        # Return
        return train_set, test_set
