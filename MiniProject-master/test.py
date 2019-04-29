import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
import cv2
import os
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
import matplotlib.pyplot as plt


rootPath = os.getcwd()

FRUITS = {0: "apple", 1: "bat", 2: "bear", 3: "bird", 4: "watermelon", 5: "camel", 6: "carrot", 7: "cat", 8: "cow",
          9: "crab", 10: "crocodile", 11: "whale", 12: "dog", 13: "duck", 14: "elephant", 15: "fish", 16: "flamingo",
          17: "frog", 18: "grapes", 19: "hedgehog", 20: "horse", 21: "kangaroo", 22: "lobster", 23: "monkey",
          24: "mosquito", 25: "mouse", 26: "onion", 27: "owl", 28: "panda", 29: "parrot", 30: "peanut", 31: "pear",
          32: "peas", 33: "penguin", 34: "pig", 35: "pineapple", 36: "pizza", 37: "rabbit", 38: "raccoon",
          39: "rhinoceros", 40: "scorpion", 41: "sea turtle", 42: "shark", 43: "sheep", 44: "snail", 45: "snake",
          46: "spider"}


def test():
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model = load_model('keras-3.h5')
    image = cv2.imread('image.png')

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    image = img_to_array(image)
    plt.imshow(image.squeeze())
    plt.savefig('img.png')

    print(image.shape)
    proba = model.predict(np.expand_dims(image, axis=0))[0]
    ind = (-proba).argsort()[:10]
    latex = [FRUITS[x] for x in ind]
    print(latex)
    f = open("fclass.txt", "w+")
    f.write(latex[0])


if __name__ == '__main__':
    # predict()
    test()
