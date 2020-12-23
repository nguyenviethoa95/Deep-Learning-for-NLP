import pandas as pd
import numpy as np
import pickle

#import matplotlib.pyplot as plt

N = 100          # length of the features vector
b_size = 10      # batch size
alpha = 0.01     # learning rate
epochs = 1000    # number of epochs

# --------------------------------LOAD DATASET--------------------------------------
####################################################################################

def convert(X):
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
        #
        split = str(X[i]).split(' ')

        for j in split:
            arr.append(float(j))

    x = np.array(arr).reshape((len(X),N))

    return x

# read the dataset from file into pandas dataframe
reviews_train = pd.read_csv(r"DATA/rt-polarity.train.vecs", sep='\t', header=None)

# Select the last column of the dataframe, which contains the vectorial representation of the text
X_train_raw = reviews_train.iloc[:, 2]
# The data is in string format so we have to format it into a matrix of integers
X_train = convert(X_train_raw)
# Select the first column of dataframe, which contains the sentiment label and map them into 0 and 1
y_train = reviews_train.iloc[:, 1].map({'label=POS': 1, 'label=NEG': 0})

reviews_dev = pd.read_csv(r"DATA/rt-polarity.dev.vecs", sep='\t', header=None)
X_dev_raw = reviews_dev.iloc[:, 2]
X_dev = convert(X_dev_raw)
y_dev = reviews_dev.iloc[:, 1].map({'label=POS': 1, 'label=NEG': 0})


reviews_test = pd.read_csv(r"DATA/rt-polarity.test.vecs", sep='\t', header=None)
X_test_raw = reviews_test.iloc[:, 2]
X_test = convert(X_test_raw)
y_test = reviews_test.iloc[:, 1].map({'label=POS': 1, 'label=NEG': 0})

#####################################################################################

# --------------------------------UTILS FUNCTION --------------------------------------
def sigmoid(z):
    """
    Compute sigmoid function given the input z

    Parameters
    ------------------------------------------
    z : array_like
    The input to the sigmoid function

    Returns
   ------------------------------------------
    g : array_like
    the computed sigmoid function. g has the same shape as z, since sigmoid is computed element-wise on z

    """
    z = np.array(z)
    g = 1 / (1 + np.exp(np.dot(-1,z)))
    return g

def grad_sigmoid(z):
    """
    Compute the derivative of sigmoid function given the input z

    Parameters
    ------------------------------------------
    z : array_like
    The input to the sigmoid function

    Returns
   ------------------------------------------
    g : array_like
    the computed sigmoid function. g has the same shape as z, since sigmoid is computed element-wise on z

    """
    g = sigmoid(z) * (1 - sigmoid(z))
    return g

def predict(theta, X):
    """
    Predict the possibility of the sample belong to the class 1
    Computes the predictions for X using the intepretation

    Parameters
    ----------------------------------------------
    theta : array_like
        Parameter for perceptron. A vector of shape (n,)
    X: array_like
        the data use for computing the predictions.
        The rows is the number of points to compute the predictions,
        and columns is the number of features

    Returns
    ---------------------------------------------
    p: array_like
        Predictions and 0 or 1 for each row in X
    """
    m,n = X.shape

    # Add intercept term to X
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    z = np.dot(X, theta)
    return sigmoid(z)

def computeSquareLoss(X,y, w):
    """
    Compute the cost with square loss and accuracy

    Parameters
    ----------------------------------------------
    y: array_like
        True label
    pred: array_like
        Predicted value of y

    Returns
    ----------------------------------------------
    L : float
        The computed value for the cost function
    """
    # perform prediction with the trained model on the dataset
    pred = sigmoid(np.dot(X, w))

    # Convert probabilities output to label (0,1)

    label = classify(pred)
    L = np.sum(np.square(y-label))

    return L

def computeLogLoss(X, y, theta):
    """
   Compute the cost with log loss

   Parameters
   ----------------------------------------------
   y: array_like
       True label
   pred: array_like
       Predicted value of y

   Returns
   ----------------------------------------------
   L : float
       The computed value for the cost function
   """
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    cost = (1/m)*(((-y).T @ np.log(h + epsilon))-((1-y).T @ np.log(1-h + epsilon)))
    return cost

def classify(y):
    """
    Assign the class label (0 or 1) to our predicted probabilities

    Parameters
    ----------------------------------------------
    theta : array_like
        Parameter for perceptron. A vector of shape (n,)
    X: array_like
        the data use for computing the predictions.
        The rows is the number of points to compute the predictions,
        and columns is the number of features

    Returns
    ---------------------------------------------
    p: array_like
        Predictions and 0 or 1 for each row in X
    """
    c = np.where(y < 0.5, 0, 1)
    return c

def computeAccuracy(y, pred):
    """
    Compute the accuracy value

    Parameters
    ----------------------------------------------
    y: array_like
        True label
    pred: array_like
        Predicted labels

    Returns
    ----------------------------------------------
    acc : float
        The computed accuracy
    """
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(y)):
        if y[i] == pred[i] == 1:
            TP += 1
        if y[i] == pred[i] == 0:
            TN += 1
        if y[i] == 0 and pred[i] == 1:
            FP += 1
        if y[i] == 1 and pred[i] == 0:
            FN += 1

    acc = (TP + TN) / (TP + TN + FP + FN)

    return acc

def updateWeights(X, y, theta, alpha,b_size):
    """ Parameters
    ----------------------------------------------
    theta : array_like
        Parameter for perceptron. A vector of shape (n,)
    X: array_like
        the data use for computing the predictions.
        The rows is the number of points to compute the predictions,
        and columns is the number of features

    Returns
    ----------------------------------------------
    """
    # cast y nparray to matrix
    y = y.reshape(-1, 1)

    # Linear hypothesis
    Z = np.dot(X, theta)

    b = (sigmoid(Z)-y)*grad_sigmoid(Z)

    c = np.ones((b_size, 1))*b

    d = np.sum(c*X, axis=0)

    theta = theta - (alpha*d/b_size).reshape(-1, 1)

    return theta


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

    # Add intercept term to X
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    # Concatenate X and y
    concat = np.concatenate((X,y.to_numpy().reshape(-1,1)),axis=1)

    arr = np.arange(m)

    shuffle = np.zeros((m, n+2))

    np.random.shuffle(arr)

    for index, i in enumerate(arr):
        shuffle[index] = concat[i]

    # Pad X with n first examples so that the number of training examples can be divided by the batch size
    pad = b_size - (X.shape[0] % b_size)

    padded = np.vstack((shuffle, shuffle[-pad:, :]))

    X,y = padded[:, 0:n+1], padded[:, -1]
    return X, y

def train(X, y, epochs, alpha, b_size):
    """
    Training the model with stochastic gradient descent

    Parameters
    ----------------------------------------------
    theta : array_like
        Parameter for perceptron. A vector of shape (n,)
    X: array_like
        the data use for computing the predictions.
        The rows is the number of points to compute the predictions,
        and columns is the number of features

    Returns
    ----------------------------------------------
    w : array_like
        The computed weights value of the model
    """
    cost_history = []

    # length of feature vector
    N = X.shape[1]

    # initialize a random weights vector
    w = np.random.normal(0, 1,(N+1, 1))

    for i in range(epochs):

        X_shuffled, y_shuffled = shuffle(X, y)
        m,n = X_shuffled.shape

        for j in range(0, m, b_size):
            w = updateWeights(X_shuffled[j:j + b_size], y_shuffled[j:j + b_size], w, alpha, b_size)

        # compute with square loss
        #cost = computeSquareLoss(X_shuffled, y_shuffled, w)

        # compute with log loss
        cost = computeLogLoss(X_shuffled, y_shuffled, w)
        cost_history.append(cost)
        print("Epoch:",i ,"\t Loss:", cost)


    # plt.plot(list(range(epochs)), cost_history, '-r')
    # plt.xlabel("Numbers of epoch ")
    # plt.ylabel("Log loss")
    # plt.suptitle(" Training log loss ")
    # plt.show()

    # test on the validation set
    pred_dev = predict(w,X_dev)
    print("Accuracy on dev set :", computeAccuracy(y_dev, classify(pred_dev)))

    # test on the test
    pred_test = predict(w, X_test)
    print("Accuracy on test set :", computeAccuracy(y_test, classify(pred_test)))


train(X_train, y_train, epochs,  alpha, b_size)
