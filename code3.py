import tensorflow as tf
import numpy as np
import pickle
import os
import cv2

path = '/home/knot/Documents/Semester6/Dl/Assignment1/cifar-10-python/cifar-10-batches-py/train_data/data_batch_1'
batch_size = 8
epochs = 3
learning_rate = 0.01
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
Y_ = tf.placeholder(tf.float32, [None, 10])
lr = tf.Variable(0.01)
drop = tf.placeholder(tf.float32)

def G2_Net():

    conv1 = tf.layers.conv2d(inputs = X, filters = 32, kernel_size = 3, padding = "same", activation = tf.nn.relu, name = "conv1")
    b_norm1 = tf.layers.batch_normalization(inputs = conv1, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "batch_norm1")
    pool1 = tf.layers.max_pooling2d(inputs = b_norm1, pool_size = 2, strides=2, name = "max_pool1")

    conv2 = tf.layers.conv2d(inputs = pool1, filters = 64, kernel_size = 3, padding = "same", activation = tf.nn.relu, name = "conv2")
    b_norm2 = tf.layers.batch_normalization(inputs = conv2, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "batch_norm2")

    conv3 = tf.layers.conv2d(inputs = b_norm2, filters = 128, kernel_size = 3, padding = "same", activation = tf.nn.relu, name = "conv3")
    b_norm3 = tf.layers.batch_normalization(inputs = conv3, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "batch_norm3")
    pool3 = tf.layers.max_pooling2d(inputs = b_norm3, pool_size = 2, strides=2, name = "max_pool3")

    pool1_flat = tf.reshape(pool3, [-1, 8 * 8 * 128])
    dense1 = tf.layers.dense(inputs = pool1_flat, units = 512, activation = tf.nn.relu, name = "dense1")
    drop1 = tf.layers.dropout(inputs = dense1,  rate = drop, noise_shape = None, seed = None, name = "dropout")
    dense2 = tf.layers.dense(inputs = drop1, units = 1024, activation = tf.nn.relu, name = "dense2")
    logits = tf.layers.dense(inputs = dense2, units = 10)
    out = tf.nn.softmax(logits, name="softmax_output")

    return out

def nn():
    prediction = G2_Net()
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = Y_))
    optimizer = tf.train.AdamOptimizer(lr).minimize(cost)
    init = tf.global_variables_initializer()
    fo = open(path, 'rb')
    dict_ = pickle.load(fo, encoding='latin1')
    label = dict_['labels']
    data = dict_['data']

    train_y = []
    for i in range(len(label)):
        temp = np.zeros(10)
        temp[label[i]] = 1
        train_y.append(temp)

    train_x = []
    for i in range(len(data)):
        img  = data[i].reshape((32,32,3))
        train_x.append(img)
    epoch = 0
    epoch_loss = 0
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):
            epoch_loss = 0
            i = 0
            while i < (len(train_x)-batch_size):
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                c = 0
                _, c, pred = sess.run([optimizer, cost, prediction], feed_dict={X: batch_x, Y_: batch_y, lr: learning_rate, drop: 0.5})
                epoch_loss += c
                i+=batch_size
            print('Epoch', epoch+1, 'completed out of',epochs,'loss:',epoch_loss)

        # correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # print('Accuracy:',accuracy.eval({x: batch_x, y:mnist.test.labels}))

def main():
    nn()



if __name__ == "__main__":
    main()
