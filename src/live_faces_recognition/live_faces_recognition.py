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
# face_cascade = cv2.CascadeClassifier('config/haarcascade_frontalface_default.xml')

face_cascade = cv2.CascadeClassifier('config/lbpcascade_frontalface_improved.xml')

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

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    incrTL = incrBL = x1 + r
    incrTR = incrBR = y1 + r

    for i in range(0, 100):
        if incrTL <= x2-8:
            cv2.line(img, (incrTL, y1), (incrTL+8, y1), color, 2)
        if incrTR <= y2-8:
            cv2.line(img, (x2, incrTR), (x2, incrTR+8), color, 2)
        if incrBL <= x2-8:
            cv2.line(img, (incrBL, y2), (incrBL+8, y2), color, 2)
        if incrBR <= y2-8:
            cv2.line(img, (x1, incrBR), (x1, incrBR+8), color, 2)

        incrTL += 16
        incrTR += 16
        incrBL += 16
        incrBR += 16

    cv2.line(img, (int(x1+((x2-x1)/2)), y1), (int(x1+((x2-x1)/2)), y1+16), color, 3) # top
    cv2.line(img, (int(x1+((x2-x1)/2)), y2), (int(x1+((x2-x1)/2)), y2-16), color, 3) # bottom
    cv2.line(img, (x1, int(y1+((y2-y1)/2))), (x1+16, int(y1+((y2-y1)/2))), color, 3) # left
    cv2.line(img, (x2, int(y1+((y2-y1)/2))), (x2-16, int(y1+((y2-y1)/2))), color, 3) # left

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

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
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            y-=70
            x-=40
            w+=100
            h+=100
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

                if line == "can you see me":
                    # draw a rectangle around the face
                    # cv2.rectangle(frame,
                    #             (x, y),  # start_point
                    #             (x+w, y+h),  # end_point
                    #             color,
                    #             2)  # thickness in px
                    draw_border(frame, (x, y), (x+w, y+h), color, 5, 5, 10)

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
