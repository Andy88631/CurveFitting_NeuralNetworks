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
seq_len    = 512
epoch      = 180
learning_rate = 0.0001
start_time = time.time()
fonts = {'family' : 'Times New Roman'}
modelSavePath = r"D:\Dropbox\MachineLearning\CurveFitting_SystemIdentification\Model_Dense\Model_Dense.ckpt"
trainFilePath = r"D:\Dropbox\MachineLearning\CurveFitting_SystemIdentification\SweepSineData"
testFilePath  = r"D:\Dropbox\MachineLearning\CurveFitting_SystemIdentification\1hrMusicData"


""" Import data """
trainFileList = os.listdir(trainFilePath)
# Input training signal
trainFlie = os.path.join(trainFilePath, trainFileList[1])
print('Training file name of x: ', trainFileList[1])
fs, x_train = wavfile.read(trainFlie)
x_train = x_train/32768.0*10

# Output training signal
trainFlie = os.path.join(trainFilePath, trainFileList[0])
print('Training file name of y: ', trainFileList[0])
fs, y_train = wavfile.read(trainFlie)
y_train = y_train/32768.0*10

# Input testing signal
testFileList = os.listdir(os.path.join(testFilePath, 'input'))
testFile = os.path.join(testFilePath, 'input', testFileList[0])
print('Testing file name of x: ', testFileList[0])
fs, x_test = wavfile.read(testFile)
x_test = x_test/32768.0*10

# Output testing signal
testFileList = os.listdir(os.path.join(testFilePath, 'output'))
testFile = os.path.join(testFilePath, 'output', testFileList[0])
print('Testing file name of y: ', testFileList[0])
fs, y_test = wavfile.read(testFile)
y_test = y_test/32768.0*10


""" Data preprocessing """
# Zero padding
lenDiff = len(x_train) - len(y_train)
if lenDiff > 0:
    y_train = np.pad(y_train, (abs(lenDiff),0), 'constant', constant_values=(0))
else:
    x_train = np.pad(x_train, (abs(lenDiff),0), 'constant', constant_values=(0))
    
# Split every single batch into one row
dlen = len(x_train)/abs(seq_len)
x_train = np.array_split(x_train, dlen)
y_train = np.array_split(y_train, dlen)

# Make validation data
x_validation = x_train[-11:-1]
del(x_train[-10:])
y_validation = y_train[-11:-1]
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
          
    LVObj      = []
    meanLV     = []
    meanRS     = []
    validRSObj = []
    trainRSObj = []
    trainDataSets = data_source.ArrayDataSource([x_train_data, y_train_data], repeats=1)
    for e in range(epoch):
        """ Training loss """
        i = 0
        for (x_input, y_input) in trainDataSets.batch_iterator(batch_size=batch_size):
            feed = {x:x_input, y:y_input, learning_rate_:learning_rate}
            trainLV, _, Predict_trainValue, True_trainOutput = sess.run([cost,optimizer,yOut,y2],
                                                                        feed_dict=feed)  
            LVObj.append(trainLV)
            i += 1
            if i % 10 == 0:
                meanLV.append(np.mean(LVObj))  # Compute the average loss value of 10 batchs
                
            """ Plot """
            print(trainLV)
            # Plot loss value
            plt.subplot(221)
            plt.cla()
            plt.subplots_adjust(top=0.925,bottom=0.06,left=0.145,right=0.93,
                                hspace=0.2,wspace=0.2)
            plt.rc('font', **fonts)
            plt.xlabel('Iteration')
            plt.ylabel('Loss value')
            plt.title('Loss value: %s' %trainLV + ' , Epoch: %s' %e)
            plt.plot(meanLV)
            plt.draw()
            plt.pause(1e-17)
            plt.grid(True)
            # Plot curves
            plt.subplot(212)
            plt.cla()
            plt.plot(Predict_trainValue[0,:], label='Predict value')
            plt.plot(True_trainOutput[0,:], color='red', label='True value')
            plt.title('Example of curves')
            plt.xlabel('Sequence length')
            plt.ylabel('Amplitude')
            plt.legend(loc='upper right')
            plt.ylim((-0.3,0.3))
            plt.draw()
            plt.pause(1e-17)   
            plt.grid(True)
            plt.show()
    
        """ Save model """
        if e % 20 == 0:
            tf.train.Saver().save(sess, modelSavePath, global_step=e)
            print('Model of No. %f epoch saved!' %e)
        
        """ Compute Training & Validation R-Square """
        validationDataSets = data_source.ArrayDataSource([x_validation_data, 
                                                          y_validation_data], repeats=1)
        for (x_input, y_input) in validationDataSets.batch_iterator(batch_size=batch_size):
            feed = {x:x_input, y:y_input, learning_rate_:learning_rate}
            validLV, _, Predict_validValue, True_validOutput = sess.run([cost,optimizer,yOut,y2],
                                                                             feed_dict=feed)  
            # R-Square (coefficient of determination)
            validRS = (np.corrcoef(Predict_validValue, True_validOutput))**2
            validRSObj.append(validRS[0,1])
#                trainRS = (np.corrcoef(True_trainOutput,Predict_validValue))**2
#                trainRSObj.append(trainRS[0,1])
        meanRS.append(np.mean(validRSObj))
        
        """ Plot """
        # Plot loss value
        plt.subplot(222)
        plt.cla()
        plt.subplots_adjust(top=0.925,bottom=0.06,left=0.145,right=0.93,
                            hspace=0.2,wspace=0.2)
        plt.rc('font', **fonts)
        plt.xlabel('Epoch')
        plt.ylabel('R-Square value')
        plt.title('Coefficient of determination')
        plt.plot(meanRS, label='Validation')
#            plt.plot(trainRSObj, label='Train')
        plt.legend(loc='upper left')
        plt.draw()
        plt.pause(1e-17)
        plt.grid(True)
        plt.show() 
    
    # Save the final model
    tf.train.Saver().save(sess, modelSavePath, global_step=e)
    
    # Count time
    training_time = time.time() - start_time
    training_time = time.strftime("%H:%M:%S", time.gmtime(training_time))
    print('Training time: ' + str(training_time))
 
    # Memory recycle
    sess.close() 

