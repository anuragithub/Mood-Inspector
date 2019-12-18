import cv2
import numpy as np

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

while cv2.waitKey(1) == -1 and not clicked:
    sucess,frame = cameraCapture.read()
    input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = np.shape(input_img)
    detected = detector(input_img)
    if len(detected)>0:
        for (x,y,w,h) in detected:
             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("Screen",frame)
cv2.destroyWindow("Screen")  
cv2.destroyAllWindows()

