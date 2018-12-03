# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 01:14:49 2018

@author: YenTa Chiang
"""

import os
from scipy.io import wavfile


#xFilePath = r'D:\Dropbox\ElectroAcoustic\Thesis\EarphoneMeasurement\SoundChick\RecordAudio\shortRecord\input'
#yFilePath = r'D:\Dropbox\ElectroAcoustic\Thesis\EarphoneMeasurement\SoundChick\RecordAudio\shortRecord\output'
#earphone      = 'ASUS'
#trainIndex    = 0
#testIndex     = 27

def ImportData(xFilePath, yFilePath, earphone, Index):
    # Input training signal
    inputPath     = os.path.join(xFilePath, earphone)
    inputFileList = os.listdir(inputPath)
    
    trainFlie     = os.path.join(inputPath, inputFileList[Index])
    fs, x = wavfile.read(trainFlie)
    x     = x/32768.0*10    # Linear Shifting
    
    # Output training signal
    outputPath    = os.path.join(yFilePath, earphone)
    outputFileList= os.listdir(outputPath)
    trainFlie     = os.path.join(outputPath, outputFileList[Index])
    _, y = wavfile.read(trainFlie)
    y    = y/32768.0*10
    
    return x, y, fs


#x_train, y_train, fs = ImportData(xFilePath, yFilePath, earphone, trainIndex)
#x_test,  y_test,  _  = ImportData(xFilePath, yFilePath, earphone, testIndex)


