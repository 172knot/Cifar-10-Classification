import tensorflow as tf
import numpy as np
import pickle
import os
import cv2

epochs = 100
batch_size = 8

path = '/home/knot/Documents/Semester6/Dl/Assignment1/cifar-10-python/cifar-10-batches-py/train_data/data_batch_1'

def convolution_neural_network():
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    Y = tf.placeholder(tf.float32, [None, 10])
    Y1 = tf.layers.conv2d(X, filters=32, kernel_size=[7, 7], strides=1, padding="same", activation=tf.nn.relu)
    Y1 = tf.layers.batch_normalization(Y1, axis = -1, momentum=0.99, epsilon=0.001)
    Y1 = tf.layers.max_pooling2d(Y1, ksize=[1,2,2,3], strides=[1,2,2,1], padding = 'same')

    dense = tf.reshape(Y1,[-1,32*32*32])
    fc1 = tf.layers.dense(Y4, 1024, activation=tf.nn.relu)
    fc2 = tf.layers.dense(fc1, 10)
    Y = tf.nn.softmax(fc2)

    return Y

def train_neural_network():
    prediction = convolution_neural_network()
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    fo = open(file, 'rb')
    dict_ = pickle.load(fo, encoding='latin1')
    label = dict_['labels']
    data = dict_['data']

    train_y = []
    for i in range(len(label)):
        temp = np.zeros([10])
        temp[label[i]] = 1
        train_y.append(temp)

    train_x = []
    for i in range(len(data)):
        img  = data[i].reshape((32,32,3))
        train_x.append(img)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(epochs):
            epoch_loss = 0
			i=0
			while i < len(train_x):
				start = i
				end = i+batch_size
				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])
				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
				epoch_loss += c
				i+=batch_size

			print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)

        # correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # print('Accuracy:',accuracy.eval({x: batch_x, y:mnist.test.labels}))


def main():
    train_neural_network()



if __name__ == "__main__":
    main()
