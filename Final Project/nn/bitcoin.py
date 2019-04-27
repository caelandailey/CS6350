print('starting up')
import warnings
warnings.filterwarnings("ignore")
import getChart
import getData
import lookback
import tensor_flow
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from IPython import get_ipython
py.init_notebook_mode(connected=True)
get_ipython().run_line_magic('matplotlib', 'inline')


class bitcoin:

    def __init__():
        print('start bitcoin up')
        
    def main():

        
        # Get data
        training_set, test_set = getData()

        # predict
        prediction = modeling(trainx, trainy, testx, testy).predict(testx)

        # Transform
        prediction_i = scaler.inverse_transform(prediction.reshape(-1, 1))
        test_y_inv = scaler.inverse_transform(testy.reshape(-1, 1))
        prediction2_i = np.array(prediction_i[:,0][1:])
        test2_y_inv = np.array(test_y_inv[:,0])

        # Train
        _, predictions = flow(data, split, modeling, rmse, n_train = 600,n_test = 60)

        # Get dif
        mean, _ = validate(data, split, modeling, rmse, flow)

        # Show chart
        getChart(test_Dates, test2_y_inv,  predictions - mean)
