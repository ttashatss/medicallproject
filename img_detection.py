import os
import cv2
import numpy as np
import tensorflow as tf
import keras.utils as image
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.utils import img_to_array 
from keras.models import  load_model
import matplotlib.pyplot as plt
import keyboard

import argparse
import sys

print('Libraries imported')

# load model
model = load_model("best_model.h5")
print('Model loaded')

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print('Ready for while loop')

while cap.isOpened():
    # print('Looping')
    ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
    if not ret:
        continue
    # test_img = cv2.flip(,1)
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)
    if cv2.waitKey(1)%256 == 32:
        cv2.imwrite('image.png', test_img)
        print('Image captured')

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.3, 5)

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
            roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
            roi_gray = cv2.resize(roi_gray, (224, 224))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            predictions = model.predict(img_pixels)

            # find max indexed array
            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]
            print(predicted_emotion)

            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        break

    # cv2.imwrite('image.png',resized_img)
    # cv2.imshow('Facial emotion analysis ', test_img)
    # cv2.waitKey(1)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows()