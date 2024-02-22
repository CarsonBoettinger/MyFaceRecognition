#Importing libraries
import threading
import cv2
from deepface import DeepFace

#Initalizing webcam parameters 
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # set Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # set Height

#Initalizing variables
count = 0
face_match = False

#Loading in reference image
ref_image = cv2.imread('photo.jpg')

#Intialize the actual machine learning function
def check_face(frame):
    global face_match
    #Using deepface to check if the face matches the reference image
    try:
        if DeepFace.verify(frame, ref_image.copy())['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        pass

#Generating loop to interate through webcam frames
while True:
    ret, frame = cap.read()
    if ret:
        #Counting over framewise intialization
        if count % 60 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(), ref_image)).start()
            except ValueError:
                pass
        count += 1
        #Checking if it is a match  
        if face_match:
            #Text on the screen 'match'
            cv2.putText(frame, "Carson", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            #Text on the screen 'no match
            cv2.putText(frame, "No match", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.imshow('Video', frame)
    #Getting the key pressed
    key = cv2.waitKey(1)
    #If 'q" is pressed, breaking the loop
    if key == ord('q'):
        break

#Killing all active windows that are open post loop    
cv2.destroyAllWindows()
    