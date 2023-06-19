from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers.legacy import Adam
from cnnmodel.smallervggnet import SmallerVGGNet
from keras.utils import to_categorical
from keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from tensorflow.keras.models import Model

INIT_LR = 1e-4
BS = 30
EPOCHS = 400
IMAGE_DIMS = (32, 32, 3)

num_classes = 10
# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Reduce pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# flatten the label values
y_train, y_test = y_train.flatten(), y_test.flatten()

K = len(set(y_train))

print("number of classes:", K)

model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
                            depth=IMAGE_DIMS[2], classes=K)

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

batch_size = 32
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

train_generator = data_generator.flow(x_train, y_train, batch_size)
steps_per_epoch = x_train.shape[0] // batch_size

r = model.fit(train_generator, validation_data=(x_test, y_test),
              steps_per_epoch=steps_per_epoch, epochs=50)


plt.plot(r.history['accuracy'], label='acc', color='red')
plt.plot(r.history['val_accuracy'], label='val_acc', color='green')
plt.legend()

scores = model.evaluate(x_test, to_categorical(y_test, num_classes), verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))

# Convert labels to one-hot encoding
# num_classes = 10
# trainY = to_categorical(trainY, num_classes)

# # # Filter the data to include only car images (label 1)
# car_indices = trainY.flatten() == 1
# x_train = trainX[car_indices]
# y_train = trainY[car_indices]

# # Adjust the data splitting to have the same batch size for logits and labels
# x_train, x_val, y_train, y_val = train_test_split(
#     x_train, y_train, test_size=0.2, random_state=42)

# # Preprocess the data
# x_train = tf.image.resize(x_train, (224, 224))
# x_val = tf.image.resize(x_val, (224, 224))
# x_test = tf.image.resize(testX, (224, 224))
# x_train = x_train / 255.0
# x_val = x_val / 255.0
# x_test = x_test / 255.0

# # Build the CNN model with increased network size and adjusted pooling
# model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
#                             depth=IMAGE_DIMS[2], classes=num_classes)

# # Compile the model with AdamW optimizer
# optimizer = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
# model.compile(optimizer=optimizer,
#               loss="categorical_crossentropy",
#               metrics=["accuracy"])

# # Train the model
# model.fit(x_train, y_train, batch_size=BS,
#           epochs=EPOCHS, validation_data=(x_val, y_val))

# # Evaluate the model on the test set
# scores = model.evaluate(x_test, to_categorical(testY, num_classes), verbose=0)
# print("Accuracy: %.2f%%" % (scores[1] * 100))

# scp -P 10023 /Users/baby/Desktop/vehicle-brand-classification/train.py nhihtt@103.130.211.150:/home/nhihtt/data/vehicle-brand-classification
