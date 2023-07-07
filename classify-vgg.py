
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import argparse
import cv2
from keras.models import load_model

# load the model
model = load_model('tl_model_v1.weights.best.hdf5')

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--image", required=True,
                help="input image need to classify")
args = vars(ap.parse_args())
# load an image from file
image = cv2.imread(args['image'])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format

image = cv2.resize(image, (224, 224))
# convert the image pixels to a numpy array
image = np.array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
# predict the probability across all output classes
yhat = model.predict(image)
# convert the probabilities to class labels
label = decode_predictions(yhat)
# retrieve the most likely result, e.g. highest probability
label = label[0][0]
# print the classification
print('%s (%.2f%%)' % (label[1], label[2]*100))