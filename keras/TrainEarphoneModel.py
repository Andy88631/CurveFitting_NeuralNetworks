# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 15:20:07 2018

@author: YenTa Chiang
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.recurrent import SimpleRNN, LSTM
import keras.optimizers as optimizer
#from keras import losses
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.io import wavfile
from DataPreprocess import Pad0AndTrim, batch_generator
import tensorflow as tf


#%% Hyper Parameters
seq_len = 512
epoch   = 10


#%% Import data
trainFilePath = r'D:\Dropbox\ElectroAcoustic\Thesis\EarphoneMeasurement\SoundChick\RecordAudio\shortRecord'
earphone      = 'ASUS'
trainIndex    = 0
testIndex     = 27

# Input training signal
inputPath     = os.path.join(trainFilePath, 'input', earphone)
inputFileList = os.listdir(inputPath)

trainFlie     = os.path.join(inputPath, inputFileList[trainIndex])
print('Training file name of x: ', inputFileList[trainIndex])
fs, x_train = wavfile.read(trainFlie)
x_train     = x_train/32768.0*1000    # Linear Shifting

# Input testing signal
testFlie  = os.path.join(inputPath, inputFileList[testIndex])
print('Testing file name of x: ',   inputFileList[testIndex])
fs, x_test = wavfile.read(testFlie)
x_test     = x_test/32768.0*1000    

# Output training signal
outputPath    = os.path.join(trainFilePath, 'output', earphone)
outputFileList= os.listdir(outputPath)
trainFlie     = os.path.join(outputPath, outputFileList[trainIndex])
print('Training file name of y: ', outputFileList[trainIndex])
_, y_train = wavfile.read(trainFlie)
y_train    = y_train/32768.0*1000

# Output testing signal
testFlie  = os.path.join(outputPath, outputFileList[testIndex])
print('Testing file name of y: ',   outputFileList[testIndex])
fs, y_test = wavfile.read(testFlie)
y_test     = y_test/32768.0*1000


#%% Data Preprocessing
x_train, y_train = Pad0AndTrim(x_train, y_train, seq_len)
x_test,  y_test  = Pad0AndTrim(x_test,  y_test,  seq_len)

# Generate Training Batch
batchTrain = batch_generator(x_train, y_train, seq_len)
batchTest  = batch_generator(x_test,  y_test,  seq_len)


#%% GPU
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.keras.backend.set_session(sess)


#%% Build Model
model = Sequential()


# Hidden Layers
model.add(Dense(units=32, kernel_initializer='normal',activation='relu', input_dim=seq_len))
model.add(Dense(units=32, kernel_initializer='normal',activation='relu'))

# Dropout
#model.add(Dropout(0.2))

# Output Layer
model.add(Dense(units=seq_len))

adam = optimizer.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#rmsprop = optimizer.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
#Nadam = optimizer.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

model.compile(loss='mean_squared_error', optimizer=adam)

print(model.summary())


#%% Train Model  
model.fit_generator(batchTrain, steps_per_epoch=2000, epochs=epoch, validation_data=batchTest, validation_steps=2000)

#%% Save Model
#import pickle
#file = open('EarphoneModel_'+earphone+'.pickle', 'wb')
#pickle.dump(model, file)
#file.close()


#%% [Training Data] Use Model to Predict & Compute R-Squsre Value
#x_pred = []
y_pred = []
for i in range(0, len(x_train), seq_len):
    x_trainForPred = x_train[i:i+seq_len]
    x_trainForPred = x_trainForPred.reshape(1,-1)
    
    # Prediction
    pred = model.predict(x_trainForPred)
    pred = pred.reshape(-1)
    y_pred.extend(pred)
    
y_pred = np.asarray(y_pred)    

# Compute R-Square Value
RS = (np.corrcoef(y_train, y_pred))**2
print('R-Square Value of Prediction in Training Data: ', RS[0,1])

#test_k = losses.mean_squared_error(x_train, y_train)
#print("原本MSE: ", K.eval(test_k))
#mse = losses.mean_squared_error(y_train, y_pred)
#print("使用模型預測後整體MSE: ", K.eval(mse))

# Plot Training Result 
t = np.linspace(0, len(y_train)/fs, len(y_train))
plt.figure(1)
plt.cla()
plt.plot(t, y_pred, label='Prediction output')
plt.plot(t, y_train, label='Output from earphone')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend(loc='upper left')
plt.title('Training Result')
plt.show()


#%% Import Model
#with open('EarphoneModel_'+earphone+'.pickle', 'rb') as modelFile:
#    model = pickle.load(modelFile)


#%% [Testing Data] Use Model to Predict & Compute R-Squsre Value
y_predTest = []
for i in range(0, len(x_test), seq_len):
    x_testForPred = x_test[i:i+seq_len]
    x_testForPred = x_testForPred.reshape(1,-1)
    
    # Prediction
    predTest = model.predict(x_testForPred)
    predTest = predTest.reshape(-1)
    y_predTest.extend(predTest)
    
y_predTest = np.asarray(y_predTest)    

# Compute R-Square Value
RS = (np.corrcoef(y_test, y_predTest))**2
print('R-Square Value of Prediction in Testing Data: ', RS[0,1])

# Plot Testing Result 
t = np.linspace(0, len(y_test)/fs, len(y_test))
plt.figure(2)
plt.cla()
plt.plot(t, y_predTest, label='Prediction output')
plt.plot(t, y_test, label='Output from earphone')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend(loc='upper left')
plt.title('Testing Result')
plt.show()

    
