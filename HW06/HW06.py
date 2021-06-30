import numpy as np
import nltk
import tensorflow.contrib.keras as keras
import tensorflow as tf

# DATA FORMAT

# 1. Reader function for LABELED DATASET
def read_labeled(filepath):
    score, first_sent,second_sen =[],[],[]
    with open(filepath,"r",encoding="utf-8") as file:
        for line in file:
            linesplit = line.split("\t")
            score.append(linesplit[0])
            first_sent.append(str(linesplit[1]).strip())
            second_sen.append(str(linesplit[2]).strip())
    return score, first_sent,second_sen

# 2. Reader function for UNLABELED DATASET
def read_unlabeled(filepath):
    first_sent,second_sen =[],[],[]
    with open (filepath,"w") as file:
        for line in file:
            linesplit = line.split("\t")
            first_sent.append(linesplit[0].strip())
            second_sen.append(linesplit[1].strip())
    return first_sent,second_sen

# 3. Writer function
def writefile(filename,scores):
    with open(filename,"w") as outfile:
        for score in scores:
            outfile.write(score+'\n')


# Embedding the sentences
import io

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for index, line in enumerate(fin):
        if index < 20000:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = np.array(list((map(float,(tokens[1:])))))
    return data

def tokenize(sen):
    return nltk.word_tokenize(sen)

def word_vec_map(tokens, dict ):
    x = []
    for token in tokens:
        if token in dict:
            x.append(dict.get(token))
        else:
            x.append(np.zeros(300))
    return x

def generate_sent_embedding(embedding):
    vec = np.zeros(300)
    for v in embedding:
        vec = vec + v
    return vec/len(embedding)

def create_sen_embedding(data):
    result= []
    for sen in data:
        token = tokenize(sen)
        embeddings = word_vec_map(token, data)
        sen_embedding = generate_sent_embedding(embeddings)
        result.append(sen_embedding)

    return result

# load data
import os
parent = os.path.abspath(os.getcwd())
train_score, train_first_sent,train_second_sen = read_labeled(parent+ r"\DATA\training-dataset.txt")
dev_score, dev_first_sent,dev_second_sen =read_labeled(parent+r"\DATA\development-dataset.txt")
test_score, test_first_sent,test_second_sen=read_labeled(parent+r"\DATA\test-hex06-dataset.txt")

# create the sentence embedding for the word
x_train_1 = np.asarray(create_sen_embedding(train_first_sent))
x_train_2 = np.asarray(create_sen_embedding(train_second_sen))
y_train = np.asarray(train_score)

x_dev_1 = np.asarray(create_sen_embedding(dev_first_sent))
x_dev_2 = np.asarray(create_sen_embedding(dev_second_sen))
y_dev = np.asarray(dev_score)

x_test_1 = np.asarray(create_sen_embedding(test_first_sent))
x_test_2 = np.asarray(create_sen_embedding(test_second_sen))
y_test = np.asarray(test_score)


# add input layer
input1 = keras.layers.Input(shape=(300,), name="input1")
input2 = keras.layers.Input(shape=(300,), name="input2")

# add concatenate layer
merged = keras.layers.Concatenate(axis=1)([input1,input2])

# add a dropout layer with prob = 0.3
x = keras.layers.Dropout(rate=0.3, name="dropout1")(merged)

# add a dense layer with 300 dimension and relu activation
x = keras.layers.Dense(300, activation=keras.activations.relu,name="dense1")(x)

# add a dropout layer with prob = 0.3
x = keras.layers.Dropout(rate=0.3, name="dropout2")(x)

# add a dense layer with 1 dimension and sigmoid activation
outputs = keras.layers.Dense(1, activation=keras.activations.sigmoid,name="output")(x)

# Initialize model
model = tf.keras.Model(inputs=[input1,input2], outputs=outputs)

# model compile with adam optimizer and mean squared error
model.compile(
    optimizer=keras.optimizers.Adam(lr=0.01),
    loss="mean_squared_error",
    metrics=[keras.metrics.mean_squared_error]
)
# train model
history = model.fit([x_train_1, x_train_2], y_train,
                    batch_size= 100,
                    epochs=300,
                    validation_data=([x_dev_1, x_dev_2], y_dev)
                    )

# evaluate on the test data
print('\n# Evaluate on test data')
results = model.evaluate(x = {"input1":x_test_1,"input2":x_test_2}, y = {"output":y_test}, batch_size=100)
print('test loss, test acc:', results)
