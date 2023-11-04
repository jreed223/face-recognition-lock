# import the necessary packages
from imutils import paths
import face_recognition
#import argparse
import pickle
import cv2
import os


# our images are located in the dataset folder
print("[INFO] start processing faces...")
imagePaths = list(paths.list_images("dataset"))
# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1,
    len(imagePaths))) #displays counter for images
    name = imagePath.split(os.path.sep)[-2] #formats name found in directory

    # load the input image and convert it from RGB (OpenCV ordering)
    # to dlib ordering (RGB)
    image = cv2.imread(imagePath) #read image
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #formats image to RGB color
    
    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input image
    boxes = face_recognition.face_locations(rgb,
    model="hog")
    
    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes) #takes image & known face
    # loop over the encodings
    
    for encoding in encodings:
    # add each encoding + name to our set of known names and
    # encodings lists
    knownEncodings.append(encoding) #list of objects
    knownNames.append(name) #lists of strings
