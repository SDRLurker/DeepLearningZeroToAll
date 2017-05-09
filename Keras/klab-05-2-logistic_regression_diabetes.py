from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np

xy = np.loadtxt('data-03-diabetes.csv', delimiter=',')
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

model = Sequential()
model.add(Dense(1, input_dim=8, activation='sigmoid'))

sgd = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.summary()
model.fit(x_data, y_data, epochs=1000)
