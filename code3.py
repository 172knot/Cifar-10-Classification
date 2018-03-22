import tensorflow as tf
import numpy as np
import pickle
import os
import cv2


tf.reset_default_graph()

logs_path = "./logs2"
path = '/home/knot/Documents/Semester6/Dl/Assignment1_/cifar-10-python/cifar-10-batches-py/train_data/data_batch_1'

batch_size = 256
epochs = 150
learning_rate = 0.001
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
Y_ = tf.placeholder(tf.float32, [None, 10])
lr = tf.Variable(0.1)
drop_1 = tf.placeholder(tf.float32)
drop_2 = tf.placeholder(tf.float32)

def G2_Net():

    conv1 = tf.layers.conv2d(inputs = X, filters = 48, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "conv1")
    b_norm1 = tf.layers.batch_normalization(inputs = conv1, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "batch_norm1")
    conv2 = tf.layers.conv2d(inputs = b_norm1, filters = 48, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "conv2")
    pool1 = tf.layers.max_pooling2d(inputs = conv2, pool_size = 2, strides=2, name = "max_pool1")
    drop1 = tf.layers.dropout(inputs = pool1,  rate = drop_1, noise_shape = None, seed = None, name = "dropout1")

    conv3 = tf.layers.conv2d(inputs = drop1, filters = 96, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "conv3")
    b_norm2 = tf.layers.batch_normalization(inputs = conv3, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "batch_norm2")
    conv4 = tf.layers.conv2d(inputs = b_norm2, filters = 96, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "conv4")
    pool2 = tf.layers.max_pooling2d(inputs = conv4, pool_size = 2, strides=2, name = "max_pool2")
    drop2 = tf.layers.dropout(inputs = pool2,  rate = drop_1, noise_shape = None, seed = None, name = "dropout2")

    conv5 = tf.layers.conv2d(inputs = drop2, filters = 192, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "conv5")
    b_norm3 = tf.layers.batch_normalization(inputs = conv5, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "batch_norm3")
    conv6 = tf.layers.conv2d(inputs = b_norm3, filters = 192, kernel_size = 3, strides = 1, padding = "same", activation = tf.nn.relu, name = "conv6")
    pool3 = tf.layers.max_pooling2d(inputs = conv6, pool_size = 2, strides=2, name = "max_pool3")
    drop3 = tf.layers.dropout(inputs = pool3,  rate = drop_1,    noise_shape = None, seed = None, name = "dropout3")

    pool1_flat = tf.reshape(drop3, [-1, 4 * 4* 192])
    dense1 = tf.layers.dense(inputs = pool1_flat, units = 512, activation = tf.nn.relu, name = "dense1")
    drop4 = tf.layers.dropout(inputs = dense1,  rate = drop_2, noise_shape = None, seed = None, name = "dropout4")
    dense2 = tf.layers.dense(inputs = drop4, units = 256, activation = tf.nn.relu, name = "dense2")
    drop5 = tf.layers.dropout(inputs = dense2,  rate = drop_2, noise_shape = None, seed = None, name = "dropout5")
    logits = tf.layers.dense(inputs = drop5, units = 10)
    out = tf.nn.softmax(logits, name="softmax_output")

    return out

def nn():
    prediction = G2_Net()
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = Y_))
    optimizer = tf.train.AdamOptimizer(lr).minimize(cost)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    tf.summary.scalar("cost", cost)
    tf.summary.scalar("accuracy", accuracy)

    summary_op = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    fo = open(path, 'rb')
    dict_ = pickle.load(fo, encoding='latin1')
    label = dict_['labels']
    data = dict_['data']

    train_y = []
    train_x = []
    for i in range(int(len(data))):
        temp1 = np.zeros(10)
        temp1[label[i]] = 1
        train_y.append(temp1)
        img = np.zeros([32,32,3])
        for j in range(len(data[i])):
            if(j<1024):
                temp = int(j/32)
                temp2 = j - (32*temp)
                img[temp][temp2][0] = data[i][j]
            elif(j<2048):
                temp = int((j-1024)/32)
                temp2 = (j-1024) - (32*temp)
                img[temp][temp2][1] = data[i][j]
            else:
                temp = int((j-2048)/32)
                temp2 = (j-2048) - (32*temp)
                img[temp][temp2][2] = data[i][j]
        if(i<20):
            cv2.imwrite("img{}.png".format(i),img)
        train_x.append(img)

    train_size = int(len(train_x)*0.8)
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):
            epoch_loss = 0
            i = 0
            while i < (train_size-batch_size):
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_x /= 255
                batch_y = np.array(train_y[start:end])
                c = 0
                _, c, pred, summary, acc = sess.run([optimizer, cost, prediction, summary_op, accuracy], feed_dict={X: batch_x, Y_: batch_y, lr: learning_rate, drop_1: 0.25, drop_2: 0.5})
                writer.add_summary(summary, epoch * batch_size + i)
                epoch_loss += c
                i+=batch_size
            print('Epoch: ',epoch+1,' completed out of ',epochs,' loss: ',epoch_loss, "Accuracy: ", acc)
        batch_x = np.array(train_x[train_size:len(train_x)])
        batch_y = np.array(train_y[train_size:len(train_x)])
        print('Validation Data Accuracy: ',accuracy.eval({X: batch_x, Y_:batch_y}))

def main():
    nn()

if __name__ == "__main__":
    main()
