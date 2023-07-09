
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import argparse
import cv2
from keras.models import load_model
from keras.preprocessing import image

# load the model
model = load_model('model.h5')

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--image", required=True,
                help="input image need to classify")
args = vars(ap.parse_args())

# load an image from file
# image = cv2.imread(args["image"])

# image = img_to_array(image)
# image = np.expand_dims(image, axis=0)

#Load flower jpeg image from local and set target size to 224 x 224
img = image.load_img(args["image"], target_size=(224,224))

#convert image to array
input_img = image.img_to_array(img)
input_img = np.expand_dims(input_img, axis=0)

reshaped_prediction = np.zeros((1, 1000))
reshaped_prediction[0, :7] = input_img

# Decode the predictions
decoded_predictions = decode_predictions(reshaped_prediction, top=5)

# Print the top predicted classes
print("Top predicted classes:")
for _, class_name, class_prob in decoded_predictions[0]:
    print(class_name, ":", class_prob)
    
# print("[INFO] classifying image...")
# proba = model.predict(input_img)
# print("[PROBA] proba", proba)
# print("[PROBA] proba[0]", proba[0])
# # convert the probabilities to class labels
# label = decode_predictions(proba, top=5)
# print("[LABEL]", label)

# # retrieve the most likely result, e.g. highest probability
# label = label[0][0]
# print("[LABEL]1", label)
# # print the classification
# print('%s (%.2f%%)' % (label[1], label[2]*100))
