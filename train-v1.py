from tensorflow.keras.models import Sequential
# action detectionimport tensorflow
from tensorflow.keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import HTML
from tensorflow.keras.preprocessing.image import ImageDataGenerator
IMAGE_SIZE = 128

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    horizontal_flip=True
)
train_generator = train_datagen.flow_from_directory(
    '/car',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    class_mode="sparse",
)

count = 0
for image_batch, label_batch in train_generator:
    #     print(label_batch)
    print(image_batch[0])
    break

class_names = list(train_generator.class_indices.keys())
print(class_names)
