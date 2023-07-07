from tensorflow.keras.layers import Input,Dense,Flatten,Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

IMAGE_SIZE = [224,224]

train_path = 'Cars Dataset/train'
test_path = 'Cars Dataset/test'

# Load pre-trained model
vgg = VGG16(input_shape=IMAGE_SIZE+[3], weights='imagenet',include_top=False)

for layer in vgg.layers:    #layer.trainable=False means we dont want to retrain those weights of layers.
    layer.trainable=False

#  Use glob to the total number of classes (audi, lamborghini and mercedes)
folders=glob('Cars Dataset/train/*')
print("number of classes", folders)

x=Flatten()(vgg.output)

prediction = Dense(len(folders),activation='softmax')(x)

model = Model(inputs=vgg.input,outputs=prediction)    # create model object

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range = 0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen=ImageDataGenerator(rescale=1./255)

bz = 32
epochs = 20

training_set=train_datagen.flow_from_directory('Cars Dataset/train',target_size=(224,224),batch_size=bz,class_mode='categorical')


test_set=test_datagen.flow_from_directory('Cars Dataset/test',target_size=(224,224),batch_size=32,class_mode='categorical')

r=model.fit(
    training_set,
    validation_data=test_set,
    epochs=epochs,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

y_pred=model.predict(test_set)

print("TEST", y_pred)