# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 15:20:07 2018

@author: YenTa Chiang
"""

from keras.models import Sequential
from keras.layers import Dense
import keras.optimizers as optimizer
#from keras import losses
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.io import wavfile
from DataPreprocess import Pad0AndTrim, batch_generator
import pickle


#%% Hyper Parameters
seq_len = 512
epoch   = 20


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
outputPath    = os.path.join(trainFilePath, 'input', earphone)
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


#%% Data Preprocessing (Zero padding & Trim data to integer times of sequence length)
x_train, y_train = Pad0AndTrim(x_train, y_train, seq_len)
x_test,  y_test  = Pad0AndTrim(x_test,  y_test,  seq_len)


#%% Build Model
model_calib = Sequential()

# Hidden Layers
model_calib.add(Dense(units=32, kernel_initializer='normal',activation='relu', input_dim=seq_len))
model_calib.add(Dense(units=32, kernel_initializer='normal',activation='relu'))

# Dropout
#model.add(Dropout(0.2))

# Output Layer
model_calib.add(Dense(units=seq_len))

adam = optimizer.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#rmsprop = optimizer.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
#Nadam = optimizer.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

model_calib.compile(loss='mean_squared_error', optimizer=adam)

#print(model_calib.summary())


#%% Import Earphone Model
with open('EarphoneModel_ASUS.pickle', 'rb') as EarphoneModel:
    model_earphone = pickle.load(EarphoneModel)


#%% Train Calibration Model
int batchNum = len(x_train)/seq_len
for e in range(epoch):
    for step in range(0, len(x_train), seq_len):
        x_trainForPred = x_train[step:step+seq_len]
        x_trainForPred = x_trainForPred.reshape(1,-1)
        
        beforeCalib = model_calib.predict(x_trainForPred)   # Earphone Calibration Model
        calibSignal = model_earphone.predict(beforeCalib)   # Earphone Simulation
    
        cost = model_calib.train_on_batch(calibSignal, x_trainForPred)
    
        if step % seq_len == 0:
            int b = step/seq_len
            print('Epoch: %s/%s Step: %s/%s train cost: %s' %(e, epoch, b, batchNum, cost))


#%% Save Model
import pickle
file = open('CalibModel_'+earphone+'.pickle', 'wb')
pickle.dump(model_calib, file)
file.close()
print('Model Saved!')


#%% [Training Data] Use Model to Predict & Compute R-Squsre Value
#x_pred = []
calibSignal_train = []
for i in range(0, len(x_train), seq_len):
    x_trainForPred = x_train[i:i+seq_len]
    x_trainForPred = x_trainForPred.reshape(1,-1)
    
    # Prediction
    pred = model_calib.predict(x_trainForPred)
    calibSignal_train_tmp = model_earphone.predict(pred)
    calibSignal_train_tmp = calibSignal_train_tmp.reshape(-1)
    calibSignal_train.extend(calibSignal_train_tmp)
    
calibSignal_train = np.asarray(calibSignal_train)

# Compute R-Square Value
RS = (np.corrcoef(y_train, calibSignal_train))**2
print('R-Square Value of Prediction in Training Data: ', RS[0,1])


# Plot Training Result 
t = np.linspace(0, len(y_train)/fs, len(y_train))
plt.figure(1)
plt.cla()
plt.plot(t, calibSignal_train, label='Prediction output')
plt.plot(t, y_train, label='Output from earphone')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend(loc='upper left')
plt.title('Training Result')
plt.show()


#%% [Testing Data] Use Model to Predict & Compute R-Squsre Value
calibSignal_test = []
for i in range(0, len(x_test), seq_len):
    x_testForPred = x_test[i:i+seq_len]
    x_testForPred = x_testForPred.reshape(1,-1)
    
    # Prediction
    pred = model_calib.predict(x_testForPred)
    calibSignal_test_tmp = model_earphone.predict(pred)
    calibSignal_test_tmp = calibSignal_test_tmp.reshape(-1)
    calibSignal_test.extend(calibSignal_test_tmp)
    
calibSignal_test = np.asarray(calibSignal_test)

# Compute R-Square Value
RS = (np.corrcoef(y_test, calibSignal_test))**2
print('R-Square Value of Prediction in Testing Data: ', RS[0,1])

# Plot Testing Result 
t = np.linspace(0, len(y_test)/fs, len(y_test))
plt.figure(2)
plt.cla()
plt.plot(t, calibSignal_test, label='Prediction output')
plt.plot(t, y_test, label='Output from earphone')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend(loc='upper left')
plt.title('Testing Result')
plt.show()

    
