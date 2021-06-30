import numpy as np
import gensim
import nltk
import string
import tensorflow as tf


# 3.1.1.a Read the review files
####################################################################################################
def load_file(file_path):
    result = []
    with open(file_path,"r") as infile:
        for line in infile:
          result.append(line)
    return result

def one_hot_label(labels):
    one_hot = []
    for label in labels:
        if label == "POS":
            one_hot.append(np.array([1,0]))
        else:
            one_hot.append(np.array([0,1]))
    return np.array(one_hot)

train_review = load_file("DATA/rt-polarity.train.reviews.txt")
r_train_label = load_file("DATA/rt-polarity.train.labels.txt")
train_label = one_hot_label(r_train_label)

dev_review = load_file("DATA/rt-polarity.dev.reviews.txt")
r_dev_label = load_file("DATA/rt-polarity.dev.labels.txt")
dev_label = one_hot_label(r_dev_label)

test_review = load_file("DATA/rt-polarity.test.reviews.txt")
r_test_label = load_file("DATA/rt-polarity.test.labels.txt")
test_label = one_hot_label(r_test_label)

# 3.1.1.b Load pre-trained Word2Vec model
####################################################################################################
model = model = gensim.models.KeyedVectors.load_word2vec_format('word2vec/vectors.bin', binary=True)
#model = gensim.models.Word2Vec.load("word2vec/vectors.bin")


def review_tokenizen(review):
    """Tokenizes the given sentences."""
    sent_text = nltk.sent_tokenize(review) # give a list of sentence
    tokens = []
    for sentence in sent_text:
        for token in nltk.word_tokenize(sentence):
            tokens.append(token)
    # remove punctuation
    tokens = list(filter(lambda token: token not in string.punctuation, tokens))
    return tokens

# 3.1.1.c Compute an embedding for a full review by averaging the word embedding for each word
####################################################################################################
def create_average_review_embedding(review_tokens):
    """ Create the review embedding with averaging method """
    review_embedding = np.zeros(300)
    for token in review_tokens:
        if token in model.vocab:
            review_embedding+=model[token]
        else:
            review_embedding += np.zeros(300)
    return review_embedding/len(review_tokens)

# 3.1.2 Compute the concatenated power mean word embedding
####################################################################################################
def create_power_means_review_embedding(review_tokens):
    """ Create the review embedding with from the arithmetic average, minimum, maximum and quadratic mean """

    review_embedding = []
    for token in review_tokens:
        if token in model.vocab:
           review_embedding.append(model[token])
        else:
            review_embedding.append(np.zeros(300))

    # reshape into array of shape (n,300)
    review_embedding = np.array(review_embedding).reshape(len(review_tokens), 300)

    avg_vec = np.mean(review_embedding, axis=0)
    min_vec = np.min(review_embedding,axis=0)
    max_vec = np.max(review_embedding,axis=0)
    mean_vec = np.std(review_embedding,axis=0)

    # Concatenating the power means to get the review embedding.
    embedding = np.concatenate((avg_vec,min_vec,max_vec,mean_vec),axis=0)
    return embedding

# Print out the averaging and concatenated embedding vectors of the first review
first_review_tokens = review_tokenizen(train_review[0])
first_avg_emb = create_average_review_embedding(first_review_tokens)
first_power_mean_emb = create_power_means_review_embedding(first_review_tokens)

print("Averaged word2vec embedding vector of the first review: \n",first_avg_emb)
print("Averaged word2vec embedding vector of the first review: \n",first_power_mean_emb)


def create_embedding(review_text):
    """ Create the array of reviews embedding"""
    #avg_emb = []
    power_mean_emb = []
    for review in review_text:
        review_tokens = review_tokenizen(review)
        #avg_emb.append(create_average_review_embedding(review_tokens))
        power_mean_emb.append(create_power_means_review_embedding(review_tokens))
    power_mean_emb = np.array(power_mean_emb)
    #avg_emb = np.array(avg_emb)
    return power_mean_emb

train_review_emb = create_embedding(train_review)
dev_review_emb = create_embedding(dev_review)
test_review_emb = create_embedding(test_review)


# 3.2 Embedding comparison
####################################################################################################
"""
Model configuration
- 2 hidden layers of dimension = 50
"""

review_input = tf.keras.layers.Input(shape=(1200,))
dense_1 = tf.keras.layers.Dense(50, activation=tf.keras.activations.relu)(review_input)
dense_2 = tf.keras.layers.Dense(50, activation=tf.keras.activations.relu)(dense_1)
output = tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax)(dense_2)
model = tf.keras.Model(review_input, output)

print(model.summary())
print("Defined the model.")

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.categorical_crossentropy, metrics=["accuracy"])
print("Compiled the model.")

# Train the model and observed the cross-entropy loss
history = model.fit(train_review_emb, train_label,
          validation_data=(dev_review_emb, dev_label),
          batch_size=100, epochs=2)

print("Trained the model.")

# # Evaluate the model on the test set
# loss, acc = model.evaluate(test_review_emb, test_label,batch_size=100)
# print("Test loss: {0}, test acc: {1}".format(loss, acc*100))
# model.save('model/model.hd5')
#
# Epoch 1/2
#
#  100/7464 [..............................] - ETA: 9s - loss: 0.8045 - acc: 0.0500
# 2200/7464 [=======>......................] - ETA: 0s - loss: 0.0674 - acc: 0.9568
# 4500/7464 [=================>............] - ETA: 0s - loss: 0.0343 - acc: 0.9789
# 6900/7464 [==========================>...] - ETA: 0s - loss: 0.0229 - acc: 0.9862
# 7464/7464 [==============================] - 0s 48us/sample - loss: 0.0214 - acc: 0.9873 - val_loss: 0.0027 - val_acc: 1.0000
# Epoch 2/2
#
#  100/7464 [..............................] - ETA: 0s - loss: 2.6675e-04 - acc: 1.0000
# 2200/7464 [=======>......................] - ETA: 0s - loss: 0.0027 - acc: 1.0000
# 4600/7464 [=================>............] - ETA: 0s - loss: 0.0021 - acc: 1.0000
# 7100/7464 [===========================>..] - ETA: 0s - loss: 0.0019 - acc: 1.0000
# 7464/7464 [==============================] - 0s 25us/sample - loss: 0.0018 - acc: 1.0000 - val_loss: 0.0022 - val_acc: 1.0000
# Trained the model.
#
#  100/1599 [>.............................] - ETA: 0s - loss: 1.4544e-07 - acc: 1.0000
# 1599/1599 [==============================] - 0s 11us/sample - loss: 3.2485e-04 - acc: 1.0000
# Test loss: 0.0003248475234318804, test acc: 100.0
