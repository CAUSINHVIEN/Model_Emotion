import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # load face detection cascade
model=load_model('model_emo.h5')
cap = cv2.VideoCapture('video.mp4') # open the camera

while True:
    ret, frame = cap.read() # read current frame from camera
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert frame to grayscale for face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5) # detect faces in frame
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 2) # draw rectangle around detected face
        face = gray[y:y+h, x:x+w] # crop image to contain only the face
        img = cv2.resize(face,(48,48))
        cv2.imwrite('faceimg.jpg',img)
        cv2.imshow('Image',img)
        test_img=load_img('faceimg.jpg',target_size=(48,48))
        test_img= img_to_array(test_img)
        test_img=test_img/255
        test_img=np.expand_dims(test_img,axis=0)
        result=model.predict(test_img)
        if   round(result[0][0])==1:
            prediction="Angry"
            print('=====> This emoj: ',prediction)
            cv2.putText(frame, prediction, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        elif round(result[0][1])==1:
            prediction="Disgust"
            print('=====> This emoj: ',prediction)
            cv2.putText(frame, prediction, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        elif round(result[0][2])==1:
            prediction="Fear"  
            print('=====> This emoj: ',prediction)
            cv2.putText(frame, prediction, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        elif round(result[0][3])==1:
            prediction="Happy"  
            print('=====> This emoj: ',prediction)
            cv2.putText(frame, prediction, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        elif round(result[0][4])==1:
            prediction="Neutral"  
            print('=====> This emoj: ',prediction)
            cv2.putText(frame, prediction, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        elif round(result[0][5])==1:
            prediction="Sad" 
            print('=====> This emoj: ',prediction)
            cv2.putText(frame, prediction, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        elif round(result[0][6])==1:
            prediction="Surprise" 
            print('=====> This emoj: ',prediction)
            cv2.putText(frame, prediction, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            print("Please show face")
    cv2.imshow('frame', frame) # show the frame with face detection rectangles
    
    if cv2.waitKey(1) == ord('q'): # quit if user presses 'q'
        break
    if cv2.waitKey(1) == ord('e'): # quit if user presses 'q'
       cv2.imwrite('faceimg2.jpg',frame)  
       print('Got It')    
cap.release() # release camera
cv2.destroyAllWindows() # close all windows