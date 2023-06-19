import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers.legacy import Adam
from cnnmodel.smallervggnet import SmallerVGGNet
from keras.utils import to_categorical

INIT_LR = 1e-4
BS = 30
EPOCHS = 400
IMAGE_DIMS = (32, 32, 3)

num_classes = 10
# Load the CIFAR-10 dataset
(trainX, trainY), (testX, testY) = keras.datasets.cifar10.load_data()


# Normalize input data to range [0, 1]
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# Convert labels to one-hot encoding
num_classes = 10
trainY = to_categorical(trainY, num_classes)

# # Filter the data to include only car images (label 1)
car_indices = trainY.flatten() == 1
x_train = trainX[car_indices]
y_train = trainY[car_indices]

# Adjust the data splitting to have the same batch size for logits and labels
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42)

# Preprocess the data
x_train = tf.image.resize(x_train, (224, 224))
x_val = tf.image.resize(x_val, (224, 224))
x_test = tf.image.resize(testX, (224, 224))
x_train = x_train / 255.0
x_val = x_val / 255.0
x_test = x_test / 255.0

# Build the CNN model with increased network size and adjusted pooling
model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
                            depth=IMAGE_DIMS[2], classes=num_classes)

# Compile the model with AdamW optimizer
optimizer = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=optimizer,
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Train the model
model.fit(x_train, y_train, batch_size=BS,
          epochs=EPOCHS, validation_data=(x_val, y_val))

# Evaluate the model on the test set
scores = model.evaluate(x_test, to_categorical(testY, num_classes), verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))

# scp -P 10023 /Users/baby/Desktop/vehicle-brand-classification/train.py nhihtt@103.130.211.150:/home/nhihtt/data/vehicle-brand-classification
