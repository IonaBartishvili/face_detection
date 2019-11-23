import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the classifiers downloaded
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Read the image and convert to grayscale format
img = cv2.imread('people.jpg')
img = cv2.resize(img, (500,400))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Calculate coordinates
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img [y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    # draw bounding boxes around detected features
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

# Plot the image

cv2.imshow('image', img)
cv2.waitKey(0)

# Write image
cv2.imwrite('people.jpg', img)