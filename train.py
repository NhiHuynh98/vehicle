import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from adamw import AdamW
from cnnmodel.smallervggnet import SmallerVGGNet

INIT_LR = 1e-4
BS = 30
EPOCHS = 400
IMAGE_DIMS = (224, 224, 3)

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Filter the data to include only car images (label 1)
car_indices = y_train.flatten() == 1
x_train = x_train[car_indices]
y_train = y_train[car_indices]

# Adjust the data splitting to have the same batch size for logits and labels
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42)

# Preprocess the data
x_train = tf.image.resize(x_train, (224, 224))
x_val = tf.image.resize(x_val, (224, 224))
x_test = tf.image.resize(x_test, (224, 224))
x_train = x_train / 255.0
x_val = x_val / 255.0
x_test = x_test / 255.0

# Build the CNN model with increased network size and adjusted pooling
model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
                            depth=IMAGE_DIMS[2], classes=10)

# Compile the model with AdamW optimizer
optimizer = AdamW(learning_rate=INIT_LR, weight_decay=0.0001)
model.compile(optimizer=optimizer,
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Train the model
model.fit(x_train, y_train, batch_size=BS,
          epochs=EPOCHS, validation_data=(x_val, y_val))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)


# scp -P 10023 /Users/baby/Desktop/vehicle-brand-classification/train.py nhihtt@103.130.211.150:/home/nhihtt/data/vehicle-brand-classification
