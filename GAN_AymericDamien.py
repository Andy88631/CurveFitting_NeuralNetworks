""" Generative Adversarial Networks (GAN).
@author: YenTa Chiang

Using generative adversarial networks (GAN) to transfer the feature of one
earphone to another.

References:
    - Generative adversarial nets. I Goodfellow, J Pouget-Abadie, M Mirza,
    B Xu, D Warde-Farley, S Ozair, Y. Bengio. Advances in neural information
    processing systems, 2672-2680.
    - Understanding the difficulty of training deep feedforward neural networks.
    X Glorot, Y Bengio. Aistats 9, 249-256

Links:
    - [GAN Paper](https://arxiv.org/pdf/1406.2661.pdf).
    - [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
    - [Xavier Glorot Init](www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.../AISTATS2010_Glorot.pdf).

Original Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from scipy.io import wavfile
import batchup.data_source as data_source

fonts = {'family' : 'Times New Roman'}

# Training Params
#num_steps = 100000
batch_size = 64
learning_rate = 0.0001
epoch = 100000

# Network Params
seq_len = 128
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = seq_len # Noise data points


""" Import data """
trainFilePath = r'D:\Dropbox\ElectroAcoustic\Thesis\EarphoneMeasurement\SoundChick\RecordAudio\Record10s'
# Input training signal
trainFileList = os.listdir(os.path.join(trainFilePath, 'output'))
trainFlie = os.path.join(trainFilePath, 'output', trainFileList[0])
print('Training file name of x: ', trainFileList[0])
fs, x_train = wavfile.read(trainFlie)
x_train = x_train/32768.0*10

# Output training signal
trainFileList = os.listdir(os.path.join(trainFilePath, 'output'))
trainFlie = os.path.join(trainFilePath, 'output', trainFileList[1])
print('Training file name of y: ', trainFileList[1])
fs, y_train = wavfile.read(trainFlie)
y_train = y_train/32768.0*10

""" Data preprocessing """
# Zero padding
lenDiff = len(x_train) - len(y_train)
if lenDiff > 0:
    y_train = np.pad(y_train, (abs(lenDiff),0), 'constant', constant_values=(0))
elif lenDiff < 0:
    x_train = np.pad(x_train, (abs(lenDiff),0), 'constant', constant_values=(0))
    
# Split every single batch into one row
dlen = np.floor(len(x_train)/abs(seq_len))
if (dlen % 2) > 0:                      # Odd to even
    dlen = dlen - 1
x_train = x_train[:int(seq_len*dlen)]   # Trim data to be multiple of seq_len
y_train = y_train[:int(seq_len*dlen)]
x_train = np.array_split(x_train, dlen)
y_train = np.array_split(y_train, dlen)

# Make validation data
validNum = int(dlen % batch_size)
x_validation = x_train[-validNum:]
del(x_train[-validNum:])
y_validation = y_train[-validNum:]
del(y_train[-validNum:])

x_train_data = np.reshape(x_train,(-1,seq_len))
y_train_data = np.reshape(y_train,(-1,seq_len))


""" A custom initialization (see Xavier Glorot init) """
def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

""" Store layers weight & bias """
weights = {
    'gen_hidden1': tf.Variable(glorot_init([noise_dim, gen_hidden_dim])),
    'gen_hidden2': tf.Variable(glorot_init([gen_hidden_dim, gen_hidden_dim*2])),
    'gen_hidden3': tf.Variable(glorot_init([gen_hidden_dim*2, gen_hidden_dim])),
    'gen_out':     tf.Variable(glorot_init([gen_hidden_dim, seq_len])),
    'disc_hidden1':tf.Variable(glorot_init([seq_len, disc_hidden_dim])),
    'disc_hidden2':tf.Variable(glorot_init([gen_hidden_dim, gen_hidden_dim*2])),
    'disc_hidden3':tf.Variable(glorot_init([gen_hidden_dim*2, gen_hidden_dim])),
    'disc_out':    tf.Variable(glorot_init([disc_hidden_dim, 1])),
}
biases = {
    'gen_hidden1': tf.Variable(tf.zeros([gen_hidden_dim])),
    'gen_hidden2': tf.Variable(tf.zeros([gen_hidden_dim*2])),
    'gen_hidden3': tf.Variable(tf.zeros([gen_hidden_dim])),
    'gen_out':     tf.Variable(tf.zeros([seq_len])),
    'disc_hidden1':tf.Variable(tf.zeros([disc_hidden_dim])),
    'disc_hidden2':tf.Variable(tf.zeros([disc_hidden_dim*2])),
    'disc_hidden3':tf.Variable(tf.zeros([disc_hidden_dim])),
    'disc_out':    tf.Variable(tf.zeros([1])),
}


""" Generator """
def generator(x):
    # Layer 1
    hidden_layer = tf.matmul(x, weights['gen_hidden1'])             # 64x128 * 128x256 = 64x256
    hidden_layer = tf.add(hidden_layer, biases['gen_hidden1'])      
    hidden_layer = tf.nn.tanh(hidden_layer)
#    hidden_layer = tf.nn.dropout(hidden_layer, dropout)
    # Normalization
    fc_mean, fc_var = tf.nn.moments(hidden_layer, axes=[0])
    scale = tf.Variable(tf.ones([1]))
    shift = tf.Variable(tf.zeros([1]))
    epsilon = 0.001
    hidden_layer = tf.nn.batch_normalization(hidden_layer, fc_mean, fc_var, shift, scale, epsilon)
    # Layer 2
    hidden_layer = tf.matmul(hidden_layer, weights['gen_hidden2'])  # 64x256 * 256x512 = 64x512
    hidden_layer = tf.add(hidden_layer, biases['gen_hidden2'])
    hidden_layer = tf.nn.tanh(hidden_layer)
#    hidden_layer = tf.nn.dropout(hidden_layer, dropout)
    # Normalization
    fc_mean, fc_var = tf.nn.moments(hidden_layer, axes=[0])
    scale = tf.Variable(tf.ones([1]))
    shift = tf.Variable(tf.zeros([1]))
    epsilon = 0.001
    hidden_layer = tf.nn.batch_normalization(hidden_layer, fc_mean, fc_var, shift, scale, epsilon)
    # Layer 3
    hidden_layer = tf.matmul(hidden_layer, weights['gen_hidden3'])  # 64x512 * 512x256 = 64x256
    hidden_layer = tf.add(hidden_layer, biases['gen_hidden3'])
    hidden_layer = tf.nn.tanh(hidden_layer)
    # Normalization
    fc_mean, fc_var = tf.nn.moments(hidden_layer, axes=[0])
    scale = tf.Variable(tf.ones([1]))
    shift = tf.Variable(tf.zeros([1]))
    epsilon = 0.001
    hidden_layer = tf.nn.batch_normalization(hidden_layer, fc_mean, fc_var, shift, scale, epsilon)
    # Fully connect
    out_layer = tf.matmul(hidden_layer, weights['gen_out'])
    out_layer = tf.add(out_layer, biases['gen_out'])
#    out_layer = tf.nn.tanh(out_layer)
    return out_layer


""" Discriminator """
def discriminator(x):
    # Layer 1
    hidden_layer = tf.matmul(x, weights['disc_hidden1'])
    hidden_layer = tf.add(hidden_layer, biases['disc_hidden1'])
    hidden_layer = tf.nn.tanh(hidden_layer)
#    hidden_layer = tf.nn.dropout(hidden_layer, dropout)
    # Normalization
    fc_mean, fc_var = tf.nn.moments(hidden_layer, axes=[0])
    scale = tf.Variable(tf.ones([1]))
    shift = tf.Variable(tf.zeros([1]))
    epsilon = 0.001
    hidden_layer = tf.nn.batch_normalization(hidden_layer, fc_mean, fc_var, shift, scale, epsilon)
    # Layer 2
    hidden_layer = tf.matmul(hidden_layer, weights['disc_hidden2'])
    hidden_layer = tf.add(hidden_layer, biases['disc_hidden2'])
    hidden_layer = tf.nn.tanh(hidden_layer)
#    hidden_layer = tf.nn.dropout(hidden_layer, dropout)
    # Normalization
    fc_mean, fc_var = tf.nn.moments(hidden_layer, axes=[0])
    scale = tf.Variable(tf.ones([1]))
    shift = tf.Variable(tf.zeros([1]))
    epsilon = 0.001
    hidden_layer = tf.nn.batch_normalization(hidden_layer, fc_mean, fc_var, shift, scale, epsilon)
    # Layer 3
    hidden_layer = tf.matmul(hidden_layer, weights['disc_hidden3'])
    hidden_layer = tf.add(hidden_layer, biases['disc_hidden3'])
    hidden_layer = tf.nn.tanh(hidden_layer)
    # Normalization
    fc_mean, fc_var = tf.nn.moments(hidden_layer, axes=[0])
    scale = tf.Variable(tf.ones([1]))
    shift = tf.Variable(tf.zeros([1]))
    epsilon = 0.001
    hidden_layer = tf.nn.batch_normalization(hidden_layer, fc_mean, fc_var, shift, scale, epsilon)
    # Fully connect
    out_layer = tf.matmul(hidden_layer, weights['disc_out'])
    out_layer = tf.add(out_layer, biases['disc_out'])
#    out_layer = tf.nn.tanh(out_layer)
    return out_layer

""" Build Networks """
# Network Inputs
gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')
disc_input= tf.placeholder(tf.float32, shape=[None, seq_len], name='disc_input')
dropout   = tf.placeholder(tf.float32)

# Build Generator Network
gen_sample = generator(gen_input)

# Build 2 Discriminator Networks (one from noise input, one from generated samples)
disc_real = discriminator(disc_input)
disc_fake = discriminator(gen_sample)

# Build Loss
gen_loss = -tf.reduce_mean(tf.log(disc_fake))
disc_loss= -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))

# Build Optimizers
optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer_disc= tf.train.AdamOptimizer(learning_rate=learning_rate)

# Training Variables for each optimizer
# By default in TensorFlow, all variables are updated by each optimizer, so we
# need to precise for each one of them the specific variables to update.
# Generator Network Variables
gen_vars = [weights['gen_hidden1'], weights['gen_hidden2'],
            weights['gen_hidden3'], weights['gen_out'],
            biases['gen_hidden1'],  biases['gen_hidden2'],
            biases['gen_hidden3'],  biases['gen_out']]
# Discriminator Network Variables
disc_vars = [weights['disc_hidden1'], weights['disc_hidden2'],
             weights['disc_hidden3'], weights['disc_out'],
             biases['disc_hidden1'],  biases['disc_hidden2'],
             biases['disc_hidden3'],  biases['disc_out']]

# Create training operations
train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_disc= optimizer_disc.minimize(disc_loss, var_list=disc_vars)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

""" Start training """
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    
    trainDataSets = data_source.ArrayDataSource([x_train_data, y_train_data], repeats=1)
    
    i = 0
    gl_append = []
    dl_append = []
#    f, a = plt.subplots(4, 4, figsize=(4, 4))
    for e in range(epoch):
        for (batch_x, batch_y) in trainDataSets.batch_iterator(batch_size=batch_size):
            i += 1
    
            # Generate noise to feed to the generator
            z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
    
            # Train
            feed_dict = {disc_input: batch_y, gen_input: batch_x, dropout: 0.4}
            _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
                                    feed_dict=feed_dict)
            gl_append.append(gl)
            dl_append.append(dl)
            if i % 100 == 0 or i == 1:
                print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))
                plt.subplot(211)
#                plt.cla()
                plt.subplots_adjust(top=0.925,bottom=0.06,left=0.145,right=0.93,
                                    hspace=0.2,wspace=0.2)
                plt.rc('font', **fonts)
                plt.xlabel('Iteration')
                plt.ylabel('Loss value')
                plt.plot(gl_append, c='#74BCFF', label='Generator')
                plt.plot(gl_append, c='#FF9359', label='Discriminator')
                plt.draw()
                plt.pause(1e-17)   
                plt.grid(True)
                plt.show()

    # Generate audio from noise, using the generator network.
    # Noise input.
    z = np.random.uniform(-1., 1., size=[4, noise_dim])
    g = sess.run([gen_sample], feed_dict={gen_input: z})
    genAudio = np.reshape(g, newshape=(4*seq_len, 1))
    
    plt.subplot(212)
    plt.plot(genAudio, label='Generator')
    plt.title('Curve from Generator ')
    plt.xlabel('Sequence length')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.ylim((-1,1))
    plt.draw()
    plt.pause(1e-17)   
    plt.grid(True)
    plt.show()