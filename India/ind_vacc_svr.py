# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('ind_vaccinations.csv')
X = dataset.iloc[0:111,5].values
y = dataset.iloc[0:111,3].values
#X=X.reshape(-1,1)
#y=y.reshape(-1,1)
#X=X.astype(np.int64)
#y=y.astype(np.int64)
#splitting dataset into test and train set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,shuffle=False)
X_train= X_train.reshape(-1,1)
X_test= X_test.reshape(-1,1)
y_train=y_train.reshape(-1,1)
y_test=y_test.reshape(-1,1)

from sklearn.preprocessing import StandardScaler
sc_X_train=StandardScaler()
sc_X_test=StandardScaler()
sc_y_train=StandardScaler()
sc_y_test=StandardScaler()
X_train=sc_X_train.fit_transform(X_train)
X_test=sc_X_test.fit_transform(X_test)
y_train=sc_y_train.fit_transform(y_train)
y_test=sc_y_test.fit_transform(y_test)

#sc_X=StandardScaler()
#sc_y=StandardScaler()
#X=sc_X.fit_transform(X)
#y=sc_y.fit_transform(y)

#fitting in model
from sklearn.svm import SVR
regressor=SVR(kernel='linear')
regressor.fit(X_train,y_train.ravel())

# Saving model to disk
pickle.dump(regressor, open('ind_svr_model.pkl','wb'))

# Saving model to disk
pickle.dump(regressor, open('ind_svr_model.pkl','wb'))

y_pred=sc_y_test.inverse_transform(regressor.predict(y_test))
y_test=sc_y_test.inverse_transform(y_test)

y_pred_day=sc_y_test.inverse_transform(regressor.predict(sc_X_train.transform(np.array([[90]]))))


import sklearn.metrics,math
mse = sklearn.metrics.mean_squared_error(y_test,y_pred)
rmse = math.sqrt(mse)
print(rmse,"\n")

#%matplotlib qt
#plt.scatter(X_test,y_test,color='red')
#plt.plot(X_test,y_pred,color='blue')