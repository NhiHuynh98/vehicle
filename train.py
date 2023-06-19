from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers.legacy import Adam
from cnnmodel.smallervggnet import SmallerVGGNet
from keras.utils import to_categorical
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow_datasets as tfds

INIT_LR = 1e-4
BS = 30
EPOCHS = 400
IMAGE_DIMS = (32, 32, 3)

num_classes = 196
# Load the CIFAR-10 dataset
# (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# car_class = 1  # The class index for car in CIFAR-10 dataset

# car_train_indices = np.where(y_train == car_class)[0]
# car_test_indices = np.where(y_test == car_class)[0]

# x_train = x_train[car_train_indices]
# y_train = y_train[car_train_indices]

# x_test = x_test[car_test_indices]
# y_test = y_test[car_test_indices]

# # Reduce pixel values

# # Reduce pixel values
# x_train, x_test = x_train / 255.0, x_test / 255.0

# # flatten the label values
# y_train, y_test = y_train.flatten(), y_test.flatten()

# print(x_train, x_test)
# print(y_train, y_test)


# Load the Cars196 dataset
dataset, info = tfds.load(
    'cars196', split=['train', 'test'], shuffle_files=True, with_info=True)

# Extract the training and testing sets
train_dataset, test_dataset = dataset['train'], dataset['test']

# Prepare the training set
x_train = []
y_train = []
for example in tfds.as_dataframe(train_dataset, info):
    x_train.append(example['image'])
    y_train.append(example['label'])

# Prepare the testing set
x_test = []
y_test = []
for example in tfds.as_dataframe(test_dataset, info):
    x_test.append(example['image'])
    y_test.append(example['label'])

# Convert the lists to NumPy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# Normalize pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# Print the shape of the loaded datasets
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)


model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
                            depth=IMAGE_DIMS[2], classes=num_classes)

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
              steps_per_epoch=steps_per_epoch, epochs=2)


plt.plot(r.history['accuracy'], label='acc', color='red')
plt.plot(r.history['val_accuracy'], label='val_acc', color='green')
plt.legend()

loss, accuracy = model.evaluate(x_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


image_number = 0

# load the image in an array
n = np.array(x_test[image_number])

# reshape it
p = n.reshape(1, 32, 32, 3)

# pass in the network for prediction and
# save the predicted label

label_names = ['Non-car', 'Car']

car_labels = np.array([label_names[int(label)] for label in y_train])

print(car_labels)

predicted_label = car_labels[model.predict(p).argmax()]

# load the original label
original_label = car_labels[y_test[image_number]]

# display the result
print("Original label is {} and predicted label is {}".format(
    original_label, predicted_label))


# y_pred = model.predict(x_test)

# threshold = 0.5  # Adjust the threshold according to your needs

# # Convert y_test to binary labels
# y_test_binary = np.where(y_test >= threshold, 1, 0)

# # Convert y_pred to binary labels
# y_pred_binary = np.where(y_pred >= threshold, 1, 0)

# Compute the confusion matrix
# cm = confusion_matrix(y_test_binary, y_pred_binary)

# sns.heatmap(cm,
#             annot=True,
#             fmt='g',
#             xticklabels=['malignant', 'benign'],
#             yticklabels=['malignant', 'benign'])
# plt.ylabel('Prediction', fontsize=13)
# plt.xlabel('Actual', fontsize=13)
# plt.title('Confusion Matrix', fontsize=17)
# plt.savefig('confusion_matrix.png')
# plt.show()


# # Finding precision and recall
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy   :", accuracy)
# precision = precision_score(y_test, y_pred)
# print("Precision :", precision)
# recall = recall_score(y_test, y_pred)
# print("Recall    :", recall)
# F1_score = f1_score(y_test, y_pred)
# print("F1-score  :", F1_score)

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
