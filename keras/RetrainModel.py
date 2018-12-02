# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 13:56:27 2018

@author: YenTa Chiang
"""

#from keras.models import Sequential
#from keras.layers import Dense
#import keras.optimizers as optimizer
#from keras import losses
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.io import wavfile
import pickle
from DataPreprocess import Pad0AndTrim, batch_generator



#%% Hyper Parameters
seq_len = 512
epoch   = 25
earphone= 'ASUS'

#%% Import Model
with open('NewParam_EarphoneModel_'+earphone+'.pickle', 'rb') as modelFile:
    model = pickle.load(modelFile)
    

#%% Import data
trainFilePath = r'D:\Dropbox\ElectroAcoustic\Thesis\EarphoneMeasurement\SoundChick\RecordAudio\shortRecord'
testIndex     = 30  # 26~32

for trainIndex in range(25):
    # Input training signal
    inputPath     = os.path.join(trainFilePath, 'input', earphone)
    inputFileList = os.listdir(inputPath)
    
    trainFlie     = os.path.join(inputPath, inputFileList[trainIndex])
    print('Training file name of x: ', inputFileList[trainIndex])
    fs, x_train = wavfile.read(trainFlie)
    x_train     = x_train/32768.0*10000    # Linear Shifting
    
    # Input testing signal
    testFlie  = os.path.join(inputPath, inputFileList[testIndex])
    print('Testing file name of x: ',   inputFileList[testIndex])
    fs, x_test = wavfile.read(testFlie)
    x_test     = x_test/32768.0*10000    # Linear Shifting
    
    # Output training signal
    outputPath    = os.path.join(trainFilePath, 'output', earphone)
    outputFileList= os.listdir(outputPath)
    trainFlie     = os.path.join(outputPath, outputFileList[trainIndex])
    print('Training file name of y: ', outputFileList[trainIndex])
    _, y_train = wavfile.read(trainFlie)
    y_train    = y_train/32768.0*10000
    
    # Output testing signal
    testFlie  = os.path.join(outputPath, outputFileList[testIndex])
    print('Testing file name of y: ',   outputFileList[testIndex])
    fs, y_test = wavfile.read(testFlie)
    y_test     = y_test/32768.0*10000    # Linear Shifting


    #%% Data Preprocessing
    x_train, y_train = Pad0AndTrim(x_train, y_train, seq_len)
    x_test,  y_test  = Pad0AndTrim(x_test,  y_test,  seq_len)

    batchTrain = batch_generator(x_train, y_train, seq_len)
    batchTest  = batch_generator(x_test,  y_test,  seq_len)
    

    #%% Train Model  
    model.fit_generator(batchTrain, steps_per_epoch=2000, epochs=epoch, validation_data=batchTest, validation_steps=2000)
 
    
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
    
    # [Testing Data] Use Model to Predict & Compute R-Squsre Value
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
    
    
    #%% Check Result
    con = input('Save model & Continue training? y/n [n]')
    if con == 'y':
        #%% Save Model
        file = open('EarphoneModel_'+earphone+'.pickle', 'wb')
        pickle.dump(model, file)
        file.close()
        print('Train next audio...')
    else:
        break
    
    
#%% Plot Training Result 
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

    
    

