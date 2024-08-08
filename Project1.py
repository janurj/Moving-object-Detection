import cv2
import imutils

cam=cv2.VideoCapture(0)
firstFrame=None
area=500
while True:
    _,img=cam.read()
    text="Normal"
    img=imutils.resize(img,width=1000)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gauss=cv2.GaussianBlur(gray,(21,21),0)
    if firstFrame is None:#run only once
        firstFrame=gauss
        continue
    imgDiff=cv2.absdiff(firstFrame,gauss)#absolute difference
    thresh=cv2.threshold(imgDiff,80,255,cv2.THRESH_BINARY)[1]
    thresh=cv2.dilate(thresh,None,iterations=2)
    cnts =cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts=imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        (x,y,w,h)=cv2.boundingRect(c)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        text="Moving Object detection"
        print(text)
    cv2.putText(img,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv2.imshow("CameraFeed",img)
    key=cv2.waitKey(10)
    print(key)
    if key==ord("f"):
        break
cam.release()
cv2.destroyAllWindows()
