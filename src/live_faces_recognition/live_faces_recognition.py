# Common imports
import numpy as np

# TensorFlow imports
# may differs from version to versions
import tensorflow as tf
from tensorflow import keras
import random
import string
import os 

# OpenCV
import cv2

# Classifier function
from src.live_faces_recognition import classifier

# opencv object that will detect faces for us
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load model to face classification
# model was created in me_not_me_classifier.ipynb notebook
use_aug = False
model_name = "face_classifier.h5"

if use_aug:
    model_name = "face_classifier_aug.h5"

model_location = 'models/'+model_name

class_names = ['admin', 'other']
imgLink = "data/train/admin/"
imgLinkAug = "data/train_aug/admin/"

def get_extended_image(img, x, y, w, h, k=0.1):
    if x - k*w > 0:
        start_x = int(x - k*w)
    else:
        start_x = x
    if y - k*h > 0:
        start_y = int(y - k*h)
    else:
        start_y = y

    end_x = int(x + (1 + k)*w)
    end_y = int(y + (1 + k)*h)

    face_image = img[start_y:end_y,
                     start_x:end_x]
    face_image = tf.image.resize(face_image, [128, 128])

    face_image = np.expand_dims(face_image, axis=0)
    return face_image

def video_capture():
    
    if os.path.isfile(model_location) != True:
        classifier.train_model(classifier.compile_model(classifier.build_model()))

    face_classifier = keras.models.load_model(model_location)

    video_capture = cv2.VideoCapture(0)  # webcamera

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=4,
            minSize=(90, 90),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:

            random_take = random.randrange(30)
            # for each face on the image detected by OpenCV
            # get extended image of this face
            face_image = get_extended_image(frame, x, y, w, h, 0.5)

            # classify face and draw a rectangle around the face
            if face_classifier:
                result = face_classifier.predict(face_image, verbose=0)
            else:
                result = [[0, 1]]

            prediction = class_names[np.array(
                result[0]).argmax(axis=0)]  # predicted class
            # confidence = np.array(result[0]).max(axis=0)  # degree of confidence

            if prediction == 'admin':
                k=0.5
                if x - k*w > 0:
                    start_x = int(x - k*w)
                else:
                    start_x = x
                if y - k*h > 0:
                    start_y = int(y - k*h)
                else:
                    start_y = y

                end_x = int(x + (1 + k)*w)
                end_y = int(y + (1 + k)*h)
                if random_take == 5:
                    # Save image in train set
                    letters = string.ascii_lowercase
                    

                    face_save = frame[start_y:end_y,
                                    start_x:end_x] # slice the face from the image
                    cv2.imwrite(imgLink+'admin-'+str(''.join(random.choice(letters) for i in range(10)))+'.jpg', face_save)
                    cv2.imwrite(imgLinkAug+'admin-'+str(''.join(random.choice(letters) for i in range(10)))+'.jpg', face_save)

                    print("Saving...")
                name = "ADMIN"
                color = (0, 250, 251)
            else:
                name = "UNKNOWN"
                color = (255, 255, 255)

            
            f = open("data/transcription.txt", "r")

            transcript = f.read().splitlines()

            for line in transcript:
                line = line.lower().strip()
                print(line.lower())

                if line == "can you see me":
                    # draw a rectangle around the face
                    cv2.rectangle(frame,
                                (x, y),  # start_point
                                (x+w, y+h),  # end_point
                                color,
                                2)  # thickness in px

                if line == "who am i":
                    cv2.putText(frame, "{:5}".format(name), (end_x-x+start_x+20, y+100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),thickness = 2)

        cv2.imshow('Video', frame)

        # Exit with ESC
        key = cv2.waitKey(1)
        if key % 256 == 27:  # ESC code
            break

    # when everything done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
