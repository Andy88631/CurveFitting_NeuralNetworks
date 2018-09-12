import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import wavfile
import batchup.data_source as data_source
import os
#from MFCC import mfcc_features
#import TFrecord_Read

""" Parameters """
batch_size = 2
seq_len = 512
start_time = time.time()
fonts = {'family' : 'Times New Roman'}
modelSavePath = r"D:\Dropbox\MachineLearning\CurveFitting_SystemIdentification\Dense_Model"


""" Import data """
filepath = r"D:\Dropbox\MachineLearning\CurveFitting_SystemIdentification\SweepSineData"
fileList = os.listdir(filepath)

# Input training signal
filename = os.path.join(filepath, fileList[1])
print('Training file name of x: ', fileList[1])
fs, x_train = wavfile.read(filename)
x_train = x_train/32768.0*10

# Output training signal
filename = os.path.join(filepath, fileList[0])
print('Training file name of y: ', fileList[0])
fs, y_train = wavfile.read(filename)
y_train = y_train/32768.0*10

# Input testing signal
inputFilePath = r"D:\Dropbox\MachineLearning\CurveFitting_SystemIdentification\1hrMusicData\input"
inputFileList = os.listdir(inputFilePath)
inputFileName = os.path.join(inputFilePath, inputFileList[0])
print('Testing file name of x: ', inputFileList[0])
fs, x_test = wavfile.read(inputFileName)
x_test = x_test/32768.0*10

# Output testing signal
outputFilePath = r"D:\Dropbox\MachineLearning\CurveFitting_SystemIdentification\1hrMusicData\output"
outputFileList = os.listdir(outputFilePath)
outputFilename = os.path.join(outputFilePath, outputFileList[0])
print('Testing file name of y: ', outputFileList[0])
fs, y_test = wavfile.read(outputFilename)
y_test = y_test/32768.0*10


""" Data preprocessing """
# Zero padding
lenDiff = len(y_train)-len(x_train)
y_train = np.pad(y_train, (abs(lenDiff),0), 'constant', constant_values=(0))

# Split every single batch into one row
dlen = len(x_train)/abs(seq_len)
x_train = np.array_split(x_train, dlen)
y_train = np.array_split(y_train, dlen)

# Make validation data
x_validation = x_train[-10:]
del(x_train[-10:])
y_validation = y_train[-10:]
del(y_train[-10:])

x_train_data = np.expand_dims(x_train, axis=2)
y_train_data = np.expand_dims(y_train, axis=2)
x_validation_data = np.expand_dims(x_validation, axis=2)
y_validation_data = np.expand_dims(y_validation, axis=2)


""" Neural networks framework """
graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32,[batch_size, seq_len, 1], name='x')
    y = tf.placeholder(tf.float32,[batch_size, seq_len, 1], name='y')
    learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')

    input= tf.reshape(x, [-1, batch_size*seq_len], name='input')
    y2   = tf.reshape(y, [-1, batch_size*seq_len], name='y2')

    yOut = tf.layers.dense(input, seq_len*batch_size, activation=tf.nn.relu, name='dense1')
#    yOut = tf.layers.dropout(yOut, 0.5)
    yOut = tf.layers.dense(yOut, 2*seq_len*batch_size, activation=tf.nn.tanh, name='dense2')
    yOut = tf.layers.dense(yOut, 2*seq_len*batch_size, activation=tf.nn.relu, name='dense3')
    yOut = tf.layers.dense(yOut, seq_len*batch_size, activation=tf.nn.tanh, name='dense4')

    """ learning """
    cost = tf.reduce_mean(tf.square(yOut-y2))
    train_op = tf.train.RMSPropOptimizer(learning_rate_)
    optimizer = train_op.minimize(cost)


""" Run session """
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True), 
                graph=graph) as sess:

    sess.run(tf.global_variables_initializer())
    
    """ Train Accuracy """    
    step    = 0
    LVObj   = []
    plotObj = []
    trainDataSets = data_source.ArrayDataSource([x_train_data, y_train_data], repeats=100)
    for (x_input, y_input) in trainDataSets.batch_iterator(batch_size=batch_size):
        LV, _, Predict_trainValue, True_trainOutput = sess.run([cost,optimizer,yOut,y2],
                                                 feed_dict={x:x_input,y:y_input,
                                                 learning_rate_:0.0001})  
        LVObj.append(LV)
        step += 1
        if step >= 10:
            plotObj.append(np.mean(LVObj))  # Compute the average loss value of 10 batchs
            
        # Plot 
        plt.figure(1)
        print(LV)
        # Plot loss value
        plt.subplot(211)
        plt.cla()
        plt.subplots_adjust(top=0.925,bottom=0.06,left=0.145,right=0.93,
                            hspace=0.2,wspace=0.2)
        plt.rc('font', **fonts)
        plt.xlabel('Iteration')
        plt.ylabel('Loss value')
        plt.title('Loss value: %s' %LV + ' , Iteration: %s' %step)
        plt.plot(plotObj)
        plt.draw()
        plt.pause(1e-17)
        plt.grid(True)
        # Plot curves
        plt.subplot(212)
        plt.cla()
        plt.plot(Predict_trainValue[0,:], label='Predict value')
        plt.plot(True_trainOutput[0,:], color='red', label='True value')
        plt.xlabel('Sequence length')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right')
        plt.ylim((-0.3,0.3))
        plt.draw()
        plt.pause(1e-17)    

        # Save model  
        if step % 10000 == 0:
            tf.train.Saver().save(sess, modelSavePath, global_step=step)

    plt.show()
    
    # Save the final model
    tf.train.Saver().save(sess, modelSavePath, global_step=step)
    
    # Count time
    training_time = time.time() - start_time
    training_time = time.strftime("%H:%M:%S", time.gmtime(training_time))
    print('Training time: ' + str(training_time))
    
    """ Validation Accuracy """
    step    = 0
    plotObj = []
    LVObj   = []
    validationDataSets = data_source.ArrayDataSource([x_validation_data, y_validation_data],
                                                     repeats=1)
    for (x_input, y_input) in validationDataSets.batch_iterator(batch_size=batch_size):
        LV, _, Predict_trainValue, True_trainOutput = sess.run([cost,optimizer,yOut,y2],
                                                         feed_dict={x:x_input,y:y_input,
                                                         learning_rate_:0.0001})  
        plotObj.append(LV)
        step += 1
#        if step >= 10:
#            plotObj.append(np.mean(LVObj))  # Compute the average loss value of 10 batchs
            
        # Plot 
        plt.figure(2)
        print(LV)
        # Plot loss value
        plt.subplot(211)
        plt.cla()
        plt.subplots_adjust(top=0.925,bottom=0.06,left=0.145,right=0.93,
                            hspace=0.2,wspace=0.2)
        plt.rc('font', **fonts)
        plt.xlabel('Iteration')
        plt.ylabel('Loss value')
        plt.title('Loss value: %s' %LV + ' , Iteration: %s' %step)
        plt.plot(plotObj)
        plt.draw()
        plt.pause(1e-17)
        plt.grid(True)
        # Plot curves
        plt.subplot(212)
        plt.cla()
        plt.plot(Predict_trainValue[0,:], label='Predict value')
        plt.plot(True_trainOutput[0,:], color='red', label='True value')
        plt.xlabel('Sequence length')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right')
        plt.ylim((-0.3,0.3))
        plt.draw()
        plt.pause(1e-17)
    
    # Memory recycle
    sess.close() 

