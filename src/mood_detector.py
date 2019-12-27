import cv2
import numpy as np
from predictor import *


def detector(frame):
    faces_rects = haar_cascade_face.detectMultiScale(frame, scaleFactor = 1.2, minNeighbors = 5);
    return faces_rects

def onMouse(event,x,y,flags,param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True

haar_cascade_face = cv2.CascadeClassifier('../model/haarcascade_frontalface_default.xml')
clicked = False
cameraCapture = cv2.VideoCapture(0)
cv2.namedWindow("Screen")
cv2.setMouseCallback("Screen",onMouse)
font = cv2.FONT_HERSHEY_SIMPLEX 
  

org = (50, 50)
fontScale = 1
color = (0, 0, 255)  
thickness = 2

sucess,frame = cameraCapture.read()
while sucess and cv2.waitKey(1) == -1 and not clicked:
    sucess,frame = cameraCapture.read()
    input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = np.shape(input_img)
    detected = detector(input_img)
    if len(detected)>0:
        #count = 0
        for (x,y,w,h) in detected:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi = frame[y:y+h,x:x+w]
            roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(roi, (350, 350))
            im_pil = Image.fromarray(resized)
            cv2.putText(frame,output(im_pil), org, font,  
                   fontScale, color, thickness, cv2.LINE_AA)
            #cv2.imwrite("roi%d.jpg" % count, resized)
            #count = count+1
    cv2.imshow("Screen",frame)
cv2.destroyWindow("Screen")  
cv2.destroyAllWindows()

