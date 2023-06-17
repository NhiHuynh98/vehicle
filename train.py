import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Filter the data to include only car images (label 1)
car_indices = y_train.flatten() == 1
x_train = x_train[car_indices]
y_train = y_train[car_indices]

# Preprocess the data
x_train = x_train.astype("float32") / 255.0
y_train = keras.utils.to_categorical(y_train, num_classes=2)

# Build the CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(2, activation="softmax")
])

# Compile the model
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2)

# Evaluate the model on the test set
x_test = x_test.astype("float32") / 255.0
y_test = keras.utils.to_categorical(y_test, num_classes=2)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)
