import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, load_img
import matplotlib.pyplot as plt
import string

train_df = pd.read_csv("C:\\Users\\sanje\\OneDrive\\Desktop\\projects\\Gestext\\sign_mnist_train.csv", delimiter=',')
test_df = pd.read_csv("C:\\Users\\sanje\\OneDrive\\Desktop\\projects\\Gestext\\sign_mnist_test.csv", delimiter=',')

print(train_df.head())


X_train, y_train = np.array(train_df.iloc[:, 1:]).reshape(-1, 28, 28).astype('float64'), np.array(train_df.label).astype('float64')
X_test, y_test = np.array(test_df.iloc[:, 1:]).reshape(-1, 28, 28).astype('float64'), np.array(test_df.label).astype('float64')
# Numpy -> Reshape
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


train_datagen = ImageDataGenerator(rescale=1.0/255.0,
                                  zoom_range=0.2,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2)
# ImageGenerator -> Keras
train_generator = train_datagen.flow(x=np.expand_dims(X_train, axis=-1), y=y_train,
                  batch_size=32)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

test_generator = test_datagen.flow(x=np.expand_dims(X_test, axis=-1), y=y_test,
                  batch_size=32) # Object instance



def plot_categories(training_images, training_labels):
  fig, axes = plt.subplots(1, 10, figsize=(16, 15))
  axes = axes.flatten()  #1-D Array
  letters = list(string.ascii_lowercase)

  for k in range(10):
    img = training_images[k]
    img = np.expand_dims(img, axis=-1)
    img = array_to_img(img) #PIL to plot easily
    ax = axes[k]
    ax.imshow(img, cmap="Greys_r")
    ax.set_title(f"{letters[int(training_labels[k])]}")
    ax.set_axis_off()


  plt.tight_layout()
  plt.show()

plot_categories(X_train, y_train)



from tensorflow.keras import Sequential #Linear Stack of layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout #Feature Extraction, downsample, Classification, prevent overfitting

tf.random.set_seed(1234)  #Random Seeds -> Same initial weights

model = tf.keras.Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),  #input and kernel
    MaxPooling2D(2, 2), #Reduce dimension - down sampling
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2), #reduces dimensionas and performs down-sampling
    Flatten(),   #Multidimensional to one dimension
    Dense(256, activation='relu'), # Rectified linear unit -> Learn complex patterns  (diff connections)
    Dropout(0.2),  #prevent Overfitting, interdependency and 0
    Dense(25, activation='softmax') #Raw scores in prob
])

model.compile(
    optimizer='adam',  #Optimization algo RMS
    loss='sparse_categorical_crossentropy', #Multiclass classification task penalizes incorrect more than correct
    metrics=['accuracy']
)

model.summary()


history = model.fit(train_generator, validation_data=test_generator, epochs=15)

model.save("sign-lang.h5")