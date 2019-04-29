from sklearn.model_selection import train_test_split as tts

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

import numpy as np
import os
from PIL import Image

# define some constants
N_FRUITS = 47
FRUITS = {0: "apple", 1: "bat", 2: "bear", 3: "bird", 4: "watermelon", 5: "camel", 6: "carrot", 7: "cat", 8: "cow",
          9: "crab", 10: "crocodile", 11: "whale", 12: "dog", 13: "duck", 14: "elephant", 15: "fish", 16: "flamingo",
          17: "frog", 18: "grapes", 19: "hedgehog", 20: "horse", 21: "kangaroo", 22: "lobster", 23: "monkey",
          24: "mosquito", 25: "mouse", 26: "onion", 27: "owl", 28: "panda", 29: "parrot", 30: "peanut", 31: "pear",
          32: "peas", 33: "penguin", 34: "pig", 35: "pineapple", 36: "pizza", 37: "rabbit", 38: "raccoon",
          39: "rhinoceros", 40: "scorpion", 41: "sea turtle", 42: "shark", 43: "sheep", 44: "snail", 45: "snake",
          46: "spider"}

# number of samples to take in each class
N = 5000

# some other constants
N_EPOCHS =10

# data files in the same order as defined in FRUITS
files = ["apple.npy", "bat.npy", "bear.npy", "bird.npy", "watermelon.npy", "camel.npy", "carrot.npy", "cat.npy",
         "cow.npy", "crab.npy", "crocodile.npy", "whale.npy", "dog.npy", "duck.npy", "elephant.npy", "fish.npy",
         "flamingo.npy", "frog.npy", "grapes.npy", "hedgehog.npy", "horse.npy", "kangaroo.npy", "lobster.npy",
         "monkey.npy", "mosquito.npy", "mouse.npy", "onion.npy", "owl.npy", "panda.npy", "parrot.npy", "peanut.npy",
         "pear.npy", "peas.npy", "penguin.npy", "pig.npy", "pineapple.npy", "pizza.npy", "rabbit.npy", "raccoon.npy",
         "rhinoceros.npy", "scorpion.npy", "sea turtle.npy", "shark.npy", "sheep.npy", "snail.npy", "snake.npy",
         "spider.npy"]


def load(dir, reshaped, files):
    "Load .npy or .npz files from disk and return them as numpy arrays. \
    Takes in a list of filenames and returns a list of numpy arrays."

    data = []
    for file in files:
        f = np.load(dir + file)
        if reshaped:
            new_f = []
            for i in range(len(f)):
                x = np.reshape(f[i], (28, 28))
                x = np.expand_dims(x, axis=0)
                x = np.reshape(f[i], (28, 28, 1))
                new_f.append(x)
            f = new_f
        data.append(f)
    return data


def normalize(data):
    "Takes a list or a list of lists and returns its normalized form"

    return np.interp(data, [0, 255], [-1, 1])


def denormalize(data):
    "Takes a list or a list of lists and returns its denormalized form"

    return np.interp(data, [-1, 1], [0, 255])




def set_limit(arrays, n):
    "Limit elements from each array up to n elements and return a single list"
    new = []
    for array in arrays:
        i = 0
        for item in array:
            if i == n:
                break
            new.append(item)
            i += 1
    return new


def make_labels(N1, N2):
    "make labels from 0 to N1, each repeated N2 times"
    labels = []
    for i in range(N1):
        labels += [i] * N2
    return labels




# images need to be flattened for training with an MLP

fruits = load("C:/Users/DELL/Desktop/data/", True, files)

fruits = set_limit(fruits, N)

# normalize the values
fruits = normalize(fruits)
print(fruits)

labels = make_labels(N_FRUITS, N)
print(labels)

# prepare the data
x_train, x_test, y_train, y_test = tts(fruits, labels, test_size=0.05)

# one hot encoding
Y_train = np_utils.to_categorical(y_train, N_FRUITS)
Y_test = np_utils.to_categorical(y_test, N_FRUITS)

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 3)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(N_FRUITS, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary)
# train
model.fit(np.array(x_train), np.array(Y_train), batch_size=32, epochs=N_EPOCHS)

print("Training complete")

print("Evaluating model")
preds = model.predict(np.array(x_test))

score = 0
for i in range(len(preds)):
    if np.argmax(preds[i]) == y_test[i]:
        score += 1

print("Accuracy: ", ((score + 0.0) / len(preds)) * 100)

model.save("doodle" + ".h5")
print("Model saved")
