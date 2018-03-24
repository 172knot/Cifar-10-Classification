import tensorflow as tf
import numpy as np
import pickle
import os


tf.reset_default_graph()

logs_path = "./logs2"
path = './data'

batch_size = 100
epochs = 50
learning_rate = 0.001
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
Y_ = tf.placeholder(tf.float32, [None, 10])
lr = tf.Variable(0.1)
drop_1 = tf.placeholder(tf.float32)
drop_2 = tf.placeholder(tf.float32)
best_acc = 0

def next_batch(i):
    fo = open(os.path.join(path,"data_batch_{}".format(i)), 'rb')
    dict_ = pickle.load(fo, encoding='latin1')
    label = dict_['labels']
    data = dict_['data']
    img = np.reshape(data,(10000,3,32,32))
    img = np.transpose(img,(0,2,3,1))

    label = np.reshape(label,(10000))
    label = label.T
    y_one_hot = np.zeros((label.size,10))
    y_one_hot[np.arange(label.size),label] = 1
    return img,y_one_hot

def test_batch():
    fo = open(os.path.join(path,"test_batch"), 'rb')
    dict_ = pickle.load(fo, encoding='latin1')
    label = dict_['labels']
    data = dict_['data']
    img = np.reshape(data,(10000,3,32,32))
    img = np.transpose(img,(0,2,3,1))

    label = np.reshape(label,(10000))
    label = label.T
    y_one_hot = np.zeros((label.size,10))
    y_one_hot[np.arange(label.size),label] = 1
    return img,y_one_hot

def G2_Net():

    conv1 = tf.layers.conv2d(inputs = X, filters = 32, kernel_size = 7, strides = 1, padding = "same", activation = tf.nn.relu, name = "conv1")
    b_norm1 = tf.layers.batch_normalization(inputs = conv1, axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True, name = "batch_norm1")
    pool1 = tf.layers.max_pooling2d(inputs = b_norm1, pool_size = 2, strides=2, name = "max_pool1")

    pool1_flat = tf.reshape(pool1, [-1, 16 * 16* 32])
    dense1 = tf.layers.dense(inputs = pool1_flat, units = 1024, activation = tf.nn.relu, name = "dense1")
    logits = tf.layers.dense(inputs = dense1, units = 10)
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
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    with tf.Session() as sess:
        saver.restore(sess,"./freeze1/model.ckpt")
        #sess.run(init)
        pt = 0
        for epoch in range(epochs):
            epoch_loss = 0
            for j in range(4):
                train_x, train_y = next_batch(j+1)
                train_size = len(train_x)
                i =0
                while i < (train_size-batch_size):
                    start = i
                    end = i+batch_size
                    batch_x = np.array(train_x[start:end])
                    batch_x = batch_x/255
                    batch_y = np.array(train_y[start:end])
                    c = 0
                    _, c, pred, summary, acc = sess.run([optimizer, cost, prediction, summary_op, accuracy], feed_dict={X: batch_x, Y_: batch_y, lr: learning_rate, drop_1: 0.25, drop_2: 0.5})
                    writer.add_summary(summary, pt)
                    pt += 1
                    epoch_loss += c
                    i+=batch_size
            save_path = saver.save(sess, "./freeze1/model.ckpt")
            batch_x, batch_y = next_batch(5)
            batch_x = batch_x/255
            sum_ = 0
            for i__ in range(0,10000,100):
                val_acc = accuracy.eval({X: batch_x[i__:i__+100], Y_:batch_y[i__:i__+100]})
                sum_ += val_acc
            sum_ = sum_/100
            print(' Epoch: ',epoch+1,' completed out of: ',epochs,' train_acc: ', acc, ' val_acc: ', sum_)
        batch_x, batch_y = test_batch()
        batch_x = batch_x/255
        sum_ = 0
        for i__ in range(0,10000,100):
            val_acc = accuracy.eval({X: batch_x[i__:i__+100], Y_:batch_y[i__:i__+100]})
            sum_ += val_acc
        sum_ = sum_/100
        print(' Test_acc: ', sum_)

def main():
    nn()

if __name__ == "__main__":
    main()
