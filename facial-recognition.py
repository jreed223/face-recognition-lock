#! /usr/bin/python
# import the necessary packages
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import pickle
import time
import cv2
import numpy as np
from urllib.request import urlopen 
from picamera2 import Picamera2, Preview
from numba import jit, njit
import RPi.GPIO  as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)


#Determine faces from encodings.pickle file model created from train_model.py
encodingsP = "encodings.pickle"

# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
#cascade = "lbpcascade_frontalcatface.xml"
cascade = "haarcascade_frontalface_default.xml"
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())
detector = cv2.CascadeClassifier(cv2.data.haarcascades + cascade)
isUnlocked = False
names = []
frameCount = 0


threshold = 65
#print(data)


def writeThingSpeak(dataList): #datalist: list of tuples [(int: fieldnum, int: data)]
    fieldData=""
    i = 0
    for item in dataList:
        i += 1 
        fieldData+="&field{}={}".format(item[0], item[1])
    
    url = "https://api.thingspeak.com/update?api_key=ZAF5WKZ4E8A03JE7"
    
    writeURL = url + fieldData
    urlopen(writeURL)

def getSimilarityScoreAvgs(currentEncoding, matchingData):
    global isUnlocked


    #face_distances = face_recognition.face_distance(data["encodings"], 
    #    currentEncoding) #calculates how different current face encoding is to known face encodings


    face_distances = face_recognition.face_distance(matchingData["encodings"], 
        currentEncoding) #calculates how different current face encoding is to known face encodings
    testedFaces = []
    average_similarity_scores = {}
    nameList = []

    for imgCounter, face_distance in enumerate(face_distances): #stores image count and face_distance val for each encoding compared to the current face
        testedEncodingName = matchingData["names"][imgCounter]
        face_similarity_score = (1/(1+face_distance))*100 #calculates score for each face encoding
        #print("The current user is a {:.2f}% match to the user {} ".format(face_similarity_score, testedEncodingName))
        writeThingSpeak([(1,face_similarity_score)])
        
        if testedEncodingName in average_similarity_scores.keys():
            average_similarity_scores[testedEncodingName] += face_similarity_score #calulates total score for each use to be averaged
        else: 
            average_similarity_scores[testedEncodingName] = face_similarity_score
            
        testedFaces.append(testedEncodingName) #adds user to the tested user list
        
        
    
  
    for knownUser in average_similarity_scores.keys():
        
        faceEncodingCount = testedFaces.count(knownUser) #counts user occurences
        average_similarity_scores[knownUser] = average_similarity_scores[knownUser]/faceEncodingCount #aerages total score by number of occurences
        score = average_similarity_scores[knownUser]
        #average_score_perUser="The current user is a {:.2f}% match to the user {} ".format(score, knownUser)
        #writeThingSpeak([(2,score)])
        if score >= threshold:
            if isUnlocked:
                authenticationResult = "USER AUHENTICATED: [{:.2f}% match to {}]".format(score, knownUser)
                authenticated = True
                nameList.append(knownUser)
                return [(authenticationResult, authenticated), (knownUser, nameList)]
            else:
                authenticationResult = "USER AUHENTICATED: Unlocking Door [{:.2f}% match to {}]".format(score, knownUser)
                authenticated = True
                nameList.append(knownUser)
                GPIO.output(18, 1)
                isUnlocked = True
                return [(authenticationResult, authenticated), (knownUser, nameList)]

        

    predictedUser = max(average_similarity_scores, key=average_similarity_scores.get)
   
    predictedUserScore = average_similarity_scores[predictedUser]
        
    if isUnlocked:
        authenticationResult = "UNAUTHORIZED USER: Locking Door [{:.2f}% match to {}]".format(predictedUserScore, predictedUser)
        authenticated = False
        GPIO.output(18, 0)
        isUnlocked = False
        return [(authenticationResult, authenticated), ("Uknown", nameList)]
    else: 
        authenticated = False
        authenticationResult = "UNAUTHORIZED USER: [{:.2f}% match to {}]".format(predictedUserScore, predictedUser)
        return [(authenticationResult, authenticated), ("Unknown", nameList)]
      

        
        
        

        
    #return (predictedUser, average_similarity_scores[predictedUser])

def getMatchingEncodings(currentEncoding, frameCount):

    # attempt to match each face in the input image to our known
    # encodings
    
    #Initialize 'currentname' to trigger only when a new person is identified.
    
    namesList = []

    displayName = "Unknown" #if face is not recognized, then print Unknown


    recognitionResults = face_recognition.compare_faces(data["encodings"], 
        currentEncoding, .5) #returns list of true/false values corresponding to matching images in the encoding
                        #True is returned if Euclidean distance < .5
    nameList = []
        
    # find the indexes of all matched faces then initialize a
    # dictionary to count the total number of times each face
    # was matched
    if True in recognitionResults:
        matchingIdxs = [i for (i, b) in enumerate(recognitionResults) if b]
        counts = {}
        encodingList = []
        # loop over the matched indexes and maintain a count for
        # each recognized face face
        for idx in matchingIdxs: 
            displayName = data["names"][idx] #binds displayName to the user corresponding with the matching encoding
            counts[displayName] = counts.get(displayName, 0) + 1 #creates dictionary with display name as key and occurence count as the value

            encodingList.append(data['encodings'][idx])
            nameList.append(data['names'][idx])
            matchingEncodings = {"encodings": encodingList, "names" : nameList}
        displayName = max(counts, key=counts.get) #selects dictionary with the largest value

        if frameCount >= 10:
            average_similarity_scores = getSimilarityScoreAvgs(currentEncoding, matchingEncodings)
            print(average_similarity_scores[0][0]) #prints results
            frameCount = 0
        #If someone in your dataset is identified, print their name on the screen
        #if currentname != displayName:
            #currentname = displayName
            #print(currentname)

    
         
        
    nameList.append(displayName)
            
    return displayName, nameList





# initialize the video stream and allow the camera sensor to warm up
# Set the ser to the followng
# src = 0 : for the build in single web cam

#vs = VideoStream(src=0,framerate=30).start()
#vs = cv2.VideoCapture(0)
cam = Picamera2()
#cam.start_preview(Preview.QTGL)
cam.start()

time.sleep(2.0) #delays loop from running before camera starts




# start the FPS counter
fps = FPS().start()

# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream and resize it
    frame = cam.capture_array()
    #frame = imutils.resize(frame, width=500)
    #frame = cv2.resize(frame,(500,300), interpolation=cv2.INTER_AREA)

    # convert the input frame from (1) BGR to grayscale (for face
    # detection) and (2) from BGR to RGB (for face recognition)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    
    
# detect faces in the grayscale frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
        minNeighbors=3, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

    # OpenCV returns bounding box coordinates in (x, y, w, h) order
    # but we need them in (top, right, bottom, left) order, so we
    # need to do a bit of reordering
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]


    # Detect the fce boxes
    #boxes = face_recognition.face_locations(frame) 
    # compute the facial embeddings for each face bounding box
    encodings = face_recognition.face_encodings(rgb, boxes)
    names  = []


    
    # loop over the facial embeddings
    frameCount+=1
    for encoding in encodings:
        #if frameCount < 3:
        matchFace = getMatchingEncodings(encoding, frameCount)
        displayName = matchFace[0]
        newNames  = matchFace[1]
        names = names + newNames



    emptyTupleCheck = np.asarray(rects)
    if emptyTupleCheck.size==0 and isUnlocked == True:
        print("No User Found: Locking Door")
        GPIO.output(18, 0)
        isUnlocked = False        
    elif emptyTupleCheck.size==0 and isUnlocked == False:
        print("No User Found")

        

    # loop over the recognized faces
    for ((top, right, bottom, left), displayName) in zip(boxes, names):
        # draw the predicted face name on the image - color is in BGR
        cv2.rectangle(rgb, (left, top), (right, bottom),
            (0, 255, 225), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(rgb, displayName, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
            .8, (0, 255, 255), 2)

    # display the image to our screen
    cv2.namedWindow("Facial Recognition is Running", cv2.WINDOW_NORMAL)
    #cv2.setWindowProperty("Facial Recognition is Running", cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Facial Recognition is Running", rgb) #displays each frame in a window
    key = cv2.waitKey(1) & 0xFF

    # quit when 'q' key is pressed
    if key == ord("q"): #breaks for loop when q is pressed
        #vs.stop()
        cam.stop()
        cv2.destroyAllWindows()
        break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
#vs.stop()
#cv2.destroyAllWindows()
#vs.stop()


