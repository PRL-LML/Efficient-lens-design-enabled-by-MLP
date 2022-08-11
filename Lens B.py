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

# Load dataset of lens B 
df = np.array(pd.read_excel('Hor_6inputs.xlsx', sheet_name='Sheet1', usecols='A:G'))

# input means number of features, ouput means number of labels
input = 6
output = 1

X = df[:,range(0,input)] 
y = df[:,range(input, input + output)]

# Split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#Scale X and y
scaler_X = MaxAbsScaler().fit(X_train)
X_train = scaler_X.transform(X_train)
X_test = scaler_X.transform(X_test)

scaler = MinMaxScaler(feature_range=(0, 1)).fit(y_train)
y_train = scaler.transform(y_train)
y_test = scaler.transform(y_test)

# Create lists to record R^2 scores and MSE losses on training and test sets after each epoch
Train_R2_list = []
Test_R2_list = []
Train_MSE_list = []
Test_MSE_list = []

a = 40  # a is number of neurons for each hidden layer
max_iter = 160 # maximum number of iterations has been pre-checked 
for i in range(1, max_iter+1):
    mlp = MLPRegressor(hidden_layer_sizes=(a,a), activation='relu', solver='adam', max_iter=i,
                       alpha=0.0001, batch_size = 128, learning_rate='constant', epsilon=1e-08, 
                       power_t=0.5, shuffle=True, random_state=0, tol=0.00001, verbose=False, 
                       warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=True,
                       validation_fraction=0.25, beta_1=0.9, beta_2=0.999,learning_rate_init=0.001,
                       n_iter_no_change=15)
    mlp.fit(X_train, y_train)
    
    # Prediction
    pre_y_train = mlp.predict(X_train)
    pre_y_test = mlp.predict(X_test)
    
    # R^2 scores and MSE loss on training and test sets
    Train_R2 = mlp.score(X_train, y_train)
    Test_R2 = mlp.score(X_test, y_test)
    Train_MSE = mean_squared_error(y_train, pre_y_train)
    Test_MSE = mean_squared_error(y_test,pre_y_test)
    
    #Add the result after each loop to the created list
    Train_R2_list.append(Train_R2)
    Test_R2_list.append(Test_R2)
    Train_MSE_list.append(Train_MSE)
    Test_MSE_list.append(Test_MSE)
    print("Interations: ", i)
    
    # Output the R^2 scores and MSE losses after the last epoch
    if i > max_iter-1:
        array1=np.array(Train_R2_list).reshape(-1,1) #train score
        array2=np.array(mlp.validation_scores_).reshape(-1,1) #validation score
        array3=np.array(Test_R2_list).reshape(-1,1) # test score   
        array4=np.array(Train_MSE_list).reshape(-1,1) #train mse
        array5=np.array(mlp.loss_curve_).reshape(-1,1) #validation mse
        array6=np.array(Test_MSE_list).reshape(-1,1) #test mse
        total = np.c_[array1, array2, array3, array4, array5, array6]
        
        ##### R^2 scores and MSE losses on training, validation, and test sets are written to an Excel
        col_1 = ('train score','validation score','test score','train mse','validation mse','test mse')
        data_1 = pd.DataFrame(total, columns = col_1)
        writer = pd.ExcelWriter('Paper_Hor_Metrics.xlsx')
        data_1.to_excel(writer, 'Sheet1', header=True, index=None, float_format='%.5g')
        writer.save()
        writer.close()

Train_R2 = mlp.score(X_train, y_train)
Test_R2 = mlp.score(X_test, y_test)
Train_MSE = mean_squared_error(pre_y_train,y_train)
Test_MSE = mean_squared_error(pre_y_test,y_test)

# print (" Train MSE: ", Train_MSE)
# print (" Test MSE:", Test_MSE)
# print(" Train Accuracy:", mlp.score(X_train, y_train))
# print(" Test Accuracy:", mlp.score(X_test, y_test))


# Predict y_steer
def prediction_steer(data):
    pred_output = mlp.predict(data).reshape(-1,1)
    final = scaler.inverse_transform(pred_output)
    return np.round(final,2)

pre_y_train_steer = prediction_steer(X_train)
pre_y_test_steer = prediction_steer(X_test)

# Actual y_steer
def inverse_transform_steer(y):
    actual_y = scaler.inverse_transform(y).reshape(-1,1)
    final = np.round(actual_y,2) 
    return final

act_y_train_steer = inverse_transform_steer(y_train)
act_y_test_steer = inverse_transform_steer(y_test)

# Write actual and predicted y values on training set to excel
Act_Pre_train = np.c_[act_y_train_steer,pre_y_train_steer]
col = ('act_y_train_steer','pre_y_train_steer')
data = pd.DataFrame(Act_Pre_train, columns = col)
writer = pd.ExcelWriter('Paper_Hor_Train_All_Act_Pre.xlsx')
data.to_excel(writer, 'Sheet1', header=True, index=None, float_format='%.5g')
writer.save()
writer.close()

# Write actual and predicted y values on test set to excel
Act_Pre_test = np.c_[act_y_test_steer,pre_y_test_steer]
col = ('act_y_test_steer','pre_y_test_steer')
data = pd.DataFrame(Act_Pre_test, columns = col)
writer = pd.ExcelWriter('Paper_Hor_Test_All_Act_Pre.xlsx')
data.to_excel(writer, 'Sheet1', header=True, index=None, float_format='%.5g')
writer.save()
writer.close()