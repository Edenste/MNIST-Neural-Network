import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load MNIST dataset
mnist = tf.keras.datasets.mnist # Training data and testing data are already separated
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalise data to 0-1
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

# Load model
model = tf.keras.models.load_model('handwritten.model')

# Test model against test data
loss, accuracy = model.evaluate(x_test, y_test)
print('Loss: ', str(loss), '\nAccuracy: ', accuracy)

# Demonstrate prediction
image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}") # Argmax returns the neuron that has the highest activation
        plt.imshow(img[0], cmap = plt.cm.binary)
        plt.show()
    except:
        print("An exception occurred")
    finally:
        image_number += 1