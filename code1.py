import tensorflow as tf
import numpy as np
import re

input_file = "input.txt"
output_file = "output.txt"

def main():
    f1 = open(input_file,'r')
    f2 = open(output_file,'w+')

    a = tf.placeholder(tf.float32,None)
    b = tf.placeholder(tf.float32,None)
    c = tf.placeholder(tf.float32,None)
    c = (a+b)*(b+1)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for t in f1.readlines():
        t__ = re.split(' |\n',t)
        f2.write(str(sess.run(c, feed_dict = {a: float(t__[0]),b: float(t__[1])})))
        f2.write("\n")
        print(sess.run(c, feed_dict = {a: float(t__[0]),b: float(t__[1])}))


    sess.close()

if __name__ == "__main__":
    main()
