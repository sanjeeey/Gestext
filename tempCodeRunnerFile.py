import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, load_img
import matplotlib.pyplot as plt
import string

train_df = pd.read_csv("C:\\Users\\sanje\\OneDrive\\Desktop\\projects\\Gestext\\sign_mnist_train.csv", delimiter=',')
test_df = pd.read_csv("C:\\Users\\sanje\\OneDrive\\Desktop\\projects\\Gestext\\sign_mnist_test.csv", delimiter=',')

print(train_df.head())