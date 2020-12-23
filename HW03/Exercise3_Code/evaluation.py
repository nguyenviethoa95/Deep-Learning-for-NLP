""" Restored save model to calculate accuracy"""

import os
import tensorflow as tf
import pandas as pd
import numpy as np


def convert(X,num_features):
    """
    Convert the each row from the data set into an array of integers

    Parameters
    ------------------------------------------
    X: array_like
        Matrix of shape (m x 1)


    Returns
   ------------------------------------------
    x : array_like
        Matrix of shape (m x n)
        M is the number of training examples and n is the number of features
    """
    arr = []
    for i in range(len(X)):
        split = str(X[i]).split(' ')

        for j in split:
            arr.append(float(j))

    x = np.array(arr).reshape((len(X),num_features))

    return x

reviews_dev = pd.read_csv(r"DATA/rt-polarity.dev.vecs", sep='\t', header=None)
X_dev_raw = reviews_dev.iloc[:, 2]
X_dev = convert(X_dev_raw,100)
y_dev = reviews_dev.iloc[:, 1].map({'label=POS': 1, 'label=NEG': 0})
BATCH_SIZE= X_dev.shape[0]


with tf.Session() as sess:
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "\model"
    new_saver = tf.train.import_meta_graph(dir_path+"/my_model_baseline/my_model_baseline.ckpt.meta")
    new_saver.restore(sess, save_path=dir_path+"/my_model_baseline/my_model_baseline.ckpt")
    graph = tf.get_default_graph()
    accuracy = graph.get_operation_by_name("Operations_/accuracy").outputs[0]

    avg_accuracy= 0
    batch_size = 10
    batch_num = 0

    limits = X_dev.shape[0] - (X_dev.shape[0] % batch_size)

    for i in range(0, limits, batch_size):
        batch_x, batch_y = X_dev[i:i + batch_size], y_dev[i:i + batch_size]
        batch_y = batch_y.to_numpy().reshape(-1,1)
        batch_accuracy = sess.run([accuracy], feed_dict={'placeholders/X:0': batch_x, 'placeholders/y:0': batch_y})
        print(batch_accuracy)
        batch_num= batch_num+1
        avg_accuracy += batch_accuracy[0]

    avg_accuracy= avg_accuracy/batch_num
    print(avg_accuracy)