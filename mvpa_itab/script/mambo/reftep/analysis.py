import h5py

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error

fname = "/media/robbis/DATA/meg/reftep/Subj1/connectivity/iPLV_and_MEP.mat"
mat = h5py.File(fname)

X = mat['iPLV'].value
y = mat['AmpsMclean'].value

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

neural_net = MLPRegressor(hidden_layer_sizes=(30), 
                          verbose=1, 
                          solver='lbfgs', 
                          max_iter=300, 
                          activation='tanh')

neural_net.fit(X_train, y_train)
y_pred = neural_net.predict(X_test)

mse = mean_squared_error(y_test, y_pred)








