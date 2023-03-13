import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

nadia = cv2.imread('../DATA/Nadia_Murad.jpg',0)
denis = cv2.imread('../DATA/Denis_Mukwege.jpg',0)

eye_cascade = cv2.CascadeClassifier('../DATA/haarcascades/haarcascade_eye.xml')

def detect_eyes(img):
    face_img = img.copy()

    eyes = eye_cascade.detectMultiScale(face_img)

    for (x, y, w, h) in eyes:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 10)

    return face_img

result = detect_eyes(nadia)
plt.imshow(result,cmap='gray')

eyes = eye_cascade.detectMultiScale(denis)

result = detect_eyes(denis)
plt.imshow(result,cmap='gray')