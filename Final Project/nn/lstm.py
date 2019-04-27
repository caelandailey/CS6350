from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from math import sqrt
from keras.callbacks import EarlyStopping
from keras.layers import Dense
import numpy as np
from keras import initializers
from keras.layers import GRU
from random import randint
from keras import initializers
from keras.layers import LSTM
import numpy as np

from sklearn.preprocessing import MinMaxScaler

class lstm:
    def __init__():
        print('started tensor_flow')
        
    def lookback(self, data, behind):
        X = list()
        Y = list()
        l = len(data)

        # Look behind us
        for i in range(l - behind):
            num = i + behind
            a = data[i:(num), 0]
            X.append(a)
            Y.append(data[num, 0])
        return np.array(X), np.array(Y)

    def validate(self, data,split,model,rmse,workflow,train = 250,test = 50,look_back):
        _list = list()

        # Validate
        for i in range(10):
            
            RE, _ = workflow(data, split, model, rmse, train, test, look_back)
            _list.append(RE)

        return np.mean(_list), _list

    def split(self, data, train, test, look_back = 1):

        start = randint(0, (len(data)-test-train))
        train_d = data[start_point:start+train]
        test_d = data[start_point+train:start+train+test]

        training_set = train_d.values
        training_set = np.reshape(training_set, (len(training_set), 1))
        test_set = test_d.values
        test_len = len(test_set)
        test_set = np.reshape(test_set, (test_len, 1))


        scal = MinMaxScaler()
        training_set = scal.fit_transform(training_set)
        test_set = scaler.transform(test_set)


        train_x, train_y = create_lookback(training_set, look_back)
        test_x, test_y = create_lookback(test_set, look_back)

        train_x_len = len(train_x)
        test_x_len = len(test_x)
        train_x = np.reshape(train_x, (train_x_len, 1, train_x.shape[1]))
        test_x = np.reshape(test_x, (test_x_len, 1, test_x.shape[1]))

        return train_x, train_y,test_x, test_y, scal, start


    def modeling(self, train_x, train_y, test_x, test_y):

        model = Sequential()
        s1 = train_x.shape[1]
        s2 = train_x.shape[2]
        model.add(GRU(256, input_shape=(s1, s2)))
        model.add(Dense(1))

        loss = 'mean_squared_error'
        opt = 'adam'
        epochs = 100
        size = 16
        model.compile(loss=loss, optimizer=opt)
        model.fit(train_x, train_y, epochs = epochs, batch_size = size, shuffle = False,
                        validation_data=(test_x, test_y), verbose=0,
                        callbacks = [EarlyStopping(monitor='val_loss',min_delta=5e-5,patience=20,verbose=0)])
        return model

    def rmse(self, model, X_test, test_y, scal, start, data, train):


        test_x = np.append(test_x, scaler.transform(np.reshape(data.iloc[start+train+len(test_x)][0], (-1,1))))
        test_x = np.reshape(test_x, (len(test_x), 1, 1))
        
        predi = model.predict(test_x)
        predi_i = scaler.inverse_transform(predi.reshape(-1, 1))
        
        test_y_i = scaler.inverse_transform(test_y.reshape(-1, 1))
        predi2_i = np.array(predi_i[:,0][1:])
        test2_y_i = np.array(test_y_i[:,0])

        RE = sqrt(mean_squared_error(test2_y_i, predi2_i))
        
        return RE, predi2_i
    
    def flow(self, data, split, model, rmse,train = 250,test = 50):
        
        train_x, train_y, test_y, test_x, scal, start = split(data, train, test)
   
        return rmse(model(train_x, train_y, test_x, test_y), X_test, Y_test, scaler, start_point, data, train)
