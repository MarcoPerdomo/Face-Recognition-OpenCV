# Face Recognition using OpenCV with webcam

# Importing the libraries
import cv2


# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #Object created from class cv2 Classifier
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') # We load the cascade for the eyes.


# Defining a function that will do the detections
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    # Will give us a tuple type with 4 elements. The first two are the coordinates on the upper left corner of the rectangle detecting the face
    # The last two elements returned are the width and height of the rectangles. 
    # First arg is the grayscale picture, second is the scaling factor, third is neighbour pixels to be accepted
    for (x, y, w, h) in faces: #coordinates x,y. Width w and height h
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
        #first arg is the frame, second is the upper left corner of rect. Third is the lower right corner of rect.
        #Fourth is color in rgb. 5th is thickness
        roi_gray = gray[y:y+h, x:x+w] #region of interest in gray
        roi_color = frame[y:y+h, x:x+w] #region of interest in gray
        smile = smile_cascade.detectMultiScale(roi_gray, 1.7, 20) 
        for (ex, ey, ew, eh) in smile: #coordinates x,y. Width w and height h
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2) #Paint a rectangle
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 15) 
        for (ax, ay, aw, ah) in eyes: #coordinates x,y. Width w and height h
            cv2.rectangle(roi_color, (ax, ay), (ax+aw, ay+ah), (0,255,255), 2) #Paint a rectangle
    return frame #Return the original image

#Doing face recognition with webcam 
video_capture = cv2.VideoCapture(0) # 0 for an internal webcam, 1 for an external device
while True:
    _, frame = video_capture.read()
    #the method returns two elements, '_,' is the last element, which is the last frame of the video
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #coloring to gray
    canvas = detect(gray, frame) #Obtaining the rectangles with the face detected
    cv2.imshow('Video', canvas) # Display the outputs
    if cv2.waitKey(1) & 0xFF == ord(' '): #Break the loop when that key is typed
        break

video_capture.release() #turnoff webcam
cv2.destroyAllWindows() #Destroy the windows