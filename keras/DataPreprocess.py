# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 15:36:29 2018

@author: YenTa Chiang
"""
import numpy as np

def Pad0AndTrim(x, y, seq_len):
    # Zero padding
    lenDiff = len(x) - len(y)
    if lenDiff > 0:
        y = np.pad(y, (abs(lenDiff),0), 'constant', constant_values=(0))
    elif lenDiff < 0:
        x = np.pad(x, (abs(lenDiff),0), 'constant', constant_values=(0))
        
    # Compute integer times of sequence length
    dlen = np.floor(len(x)/abs(seq_len))
    
    # Odd to even
    if (dlen % 2) > 0:
        dlen = dlen - 1
    
    # Trim data
    x = x[:int(seq_len*dlen)]
    y = y[:int(seq_len*dlen)]
    
    return x, y


# Generate Training Batch
def batch_generator(X, Y, seq_len):  
    while(True):    
        for i in range(0,len(X),seq_len):
            
            x_batch = X[i:i+seq_len]
            y_batch = Y[i:i+seq_len]
            
            x_batch = x_batch.reshape(1,-1)
            y_batch = y_batch.reshape(1,-1)
            
            yield x_batch, y_batch
            
            