import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt

# Plot
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR,"images",fig_id+".png")
    print("saving figure",fig_id)
    if tight_layout:
        plt.tight_layout()

    plt.savefig(path, format="png",dpi=300)

# Make the output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# --------------------------------LOAD DATASET--------------------------------------
####################################################################################

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

def shuffle(X, y):
    """
    Shuffle the row of the X data matrix

    Parameters
    ----------------------------------------------
    X: array_like
        the data use for computing the predictions.
        The rows is the number of points to compute the predictions,
        and columns is the number of features (m,100)

    y: array_like true label of size (m,1)

    Returns
    ----------------------------------------------
    """

    # Setup the data matrix appropriately, and add ones for the intercept term
    m, n = X.shape

    # Concatenate X and y
    concat = np.concatenate((X,y.to_numpy().reshape(-1,1)),axis=1)

    arr = np.arange(m)

    shuffle = np.zeros((m, n+1))

    np.random.shuffle(arr)

    for index, i in enumerate(arr):
        shuffle[index] = concat[i]

    # Pad X with n first examples so that the number of training examples can be divided by the batch size
    pad = batch_size - (X.shape[0] % batch_size)

    padded = np.vstack((shuffle, shuffle[-pad:, :]))

    X,y = padded[:, 0:n], padded[:, -1]
    return X, y

# read the dataset from file into pandas dataframe
reviews_train = pd.read_csv(r"DATA/rt-polarity.train.vecs", sep='\t', header=None)

# Select the last column of the dataframe, which contains the vectorial representation of the text
X_train_raw = reviews_train.iloc[:, 2]
# The data is in string format so we have to format it into a matrix of integers
X_train = convert(X_train_raw,100)
# Select the first column of dataframe, which contains the sentiment label and map them into 0 and 1
y_train = reviews_train.iloc[:, 1].map({'label=POS': 1, 'label=NEG': 0})


reviews_dev = pd.read_csv(r"DATA/rt-polarity.dev.vecs", sep='\t', header=None)
X_dev_raw = reviews_dev.iloc[:, 2]
X_dev = convert(X_dev_raw,100)
y_dev = reviews_dev.iloc[:, 1].map({'label=POS': 1, 'label=NEG': 0})


#####################################################################################


reset_graph()
# Training Session Parameters
n_epochs = 20  # number of epoch
alpha = 0.01  # learning rate

batch_size = 10  # batch size
n_batches = int(np.ceil(X_train.shape[0]/ batch_size))

# Features of the dataset
numFeatures = X_train.shape[1]
n_hidden_1 = 50   # number of neurons in 1st layer
n_hidden_2 = 50   # number of neurons in 1st layer
n_input = 100    # length of the input vector
n_classes = 2

# Placeholders
# tf Graph input
with tf.name_scope('placeholders'):
    X = tf.placeholder(tf.float32, shape=(batch_size, n_input), name="X")
    y = tf.placeholder(tf.float32, shape=(batch_size, 1), name="y")

# Specifying weight and bias for the first layer #/np.sqrt(n_input))
W1 = tf.Variable(tf.truncated_normal([n_input, n_hidden_1], mean= 0, stddev=1),name="weights1")
b1 = tf.Variable(tf.truncated_normal([n_hidden_1], mean = 0, stddev=1 / np.sqrt(n_input)),name="bias1")

# Adding an activation function for the first layer (relu)
y1 = tf.nn.relu((tf.matmul(X,W1)+b1), name ="activationLayer1")

# Specifying weight and bias for the second layer #, stddev=1/np.sqrt(n_input)
W2 = tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], mean = 0),name="weights2")
b2 = tf.Variable(tf.random_normal([n_hidden_2],mean=0,stddev=1/np.sqrt(n_input)),name='biases2')

# Adding an activation function for the first layer (tanh)
y2 = tf.nn.tanh((tf.matmul(y1,W2)+b2), name ="activationLayer2")

# Specifying weight and bias for the output layer #/np.sqrt(n_input)
Wo = tf.Variable(tf.truncated_normal([n_hidden_2, 1], mean = 0, stddev=1),name="weightso")
bo = tf.Variable(tf.random_normal([1], mean=0, stddev=1/np.sqrt(n_input)), name='biasesOut')

# Adding an activation function for the output layer (sigmoid)
yo = tf.nn.sigmoid((tf.matmul(y2,Wo)+bo), name ="activationOutputLayer")

with tf.name_scope('Operations_'):
# Defining the loss function
    mse_loss = tf.losses.mean_squared_error(y,yo)

    # Normal loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y))
    # Loss function with L2 Regularization with beta=0.01
    regularizers = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(Wo)
    loss = tf.reduce_mean(loss + 0.001 * regularizers)

    # Optimizer
    sdg_optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(mse_loss)
    adam_optimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(mse_loss)
    adaGrad_optimizer = tf.train.AdagradOptimizer(learning_rate=alpha).minimize(mse_loss)

    # accuracy for the test set
    accuracy = tf.reduce_mean(tf.square(tf.subtract(y,yo)),name="accuracy")

#accuracy, update_op = tf.metrics.accuracy(labels=y, predictions=tf.argmax(tf.sigmoid(output), axis=1))

# Initialization
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Training
with tf.Session() as mysess:
    mysess.run(init)
    total_batch = int(len(y_train)/batch_size)

    cost_history = []
    accuracy_history =[]

    for epoch in range(n_epochs):

        X_shuffled,y_shuffled = shuffle(X_train,y_train)

        for i in range(0, X.shape[0], batch_size):
            batch_x, batch_y = X_shuffled[i:i + batch_size], y_shuffled[i:i + batch_size]
            #print(batch_x.shape,batch_y.shape)
            batch_x = np.float32(np.reshape(batch_x,(batch_size,n_input)))
            batch_y = np.float32(np.reshape(batch_y, (batch_size,1)))

            _, c = mysess.run([sdg_optimizer, mse_loss], feed_dict={X: batch_x, y: batch_y})
            cost_history.append(c)

        # compute the average cost
    avg_loss= sum(cost_history) /n_epochs
    print(avg_loss)

    # Create a saver node to save the model
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "\model"
    save_path = saver.save(sess =mysess, save_path=dir_path+"\my_model_baseline\my_model_baseline.ckpt")
    print("Model saved in path ", str(save_path))
