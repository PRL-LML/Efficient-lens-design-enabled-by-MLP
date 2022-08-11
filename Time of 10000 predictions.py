import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from itertools import product
from sklearn import svm
from matplotlib import rcParams
import matplotlib.pyplot as plt
import time 
import warnings 
warnings.filterwarnings('ignore')

#Load dataset of lens A
Lens = np.array(pd.read_excel('Lens A dataset.xlsx', sheet_name='Sheet1', usecols='A:H'))

input = 6
output = 2

# Split training and test sets
X = Lens[:,range(0,input)] 
y = Lens[:,range(input, input + output)]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#Scale X and y
scalerX = MaxAbsScaler().fit(X_train)
X_train = scalerX.transform(X_train)
X_test = scalerX.transform(X_test)

scaler = MinMaxScaler(feature_range=(0, 1)).fit(y_train)
y_train = scaler.transform(y_train)
y_test = scaler.transform(y_test)

#Train the network
a=40
mlp = MLPRegressor(hidden_layer_sizes=(a,a), activation='relu', solver='adam', max_iter=174,
                       alpha=0.0001, batch_size = 128, learning_rate='constant', epsilon=1e-08, 
                       power_t=0.5, shuffle=True, random_state=0, tol=0.00001, verbose=False, 
                       warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=True,
                       validation_fraction=0.25, beta_1=0.9, beta_2=0.999,learning_rate_init=0.001,
                       n_iter_no_change=15)
mlp.fit(X_train, y_train)


#Predict y and show corresponding X
def prediction(data):
    fina_X = np.array(data).reshape(1,-1)
    user_input = scalerX.transform(fina_X)  
    pred_output = mlp.predict(user_input)
    acutal_y = scaler.inverse_transform(pred_output)
    final_y = np.round(acutal_y,2)
    final = np.concatenate((fina_X, final_y), axis=1)
    return final 

#Time of 13690 predictions
start =time.perf_counter()
for a in np.linspace(4,6,num=3):
    for b in np.linspace(4,6,num=3):
        for c in np.linspace(-1,1,num=3):
            for d in np.linspace(-1,1,num=3):
                for e in np.arange(0.05,0.13125,0.00625):
                    for f in np.arange(0.05,0.13125,0.00625):
                        start3=time.perf_counter()
                        output = prediction([a,b,c,d,e,f])
                        end3=time.perf_counter()
#                         print(' Distance   Thickness   CC_AF    CC_AR    C_AF   C_AR   1/e2       FWHM\n',output)
#                         print(' Each prediction time: %s s'%(end3-start3))                       
end=time.perf_counter()

#Time of 10000 predictions
time_10000= (end-start)/13690*10000
print('Time for 10000 predictions: %s s'%(time_10000))
