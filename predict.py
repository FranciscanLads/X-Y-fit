import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
x_new = [8]    # NOTE this should be a list

def predicted_answer(X, Y, x_new):
    model = Sequential()
    model.add(Dense(5, input_shape=(1,)))
    model.add(Dense(1))
    model.compile(optimizer='RMSprop', loss='mean_squared_error')
    model.fit(X, Y, epochs=500)
    return (model.predict(x_new)[0])

X = np.array([1,2,3,4,5,6,7]) #  dependant variables
Y = np.array([2,4,6,8,10,12,14]) #  independant variables
print(predicted_answer(X,Y,x_new))
