__author__ = 'Bobby'
import cv2
import numpy as np
import winsound
fclass=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def extract(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=fclass.detectMultiScale(gray,1.3,5)

    if faces is ():
        return None
    for (x,y,w,h) in faces:
        crface=img[y:y+h,x:x+w]
    return crface

capture=cv2.VideoCapture(0)
x=0

while True:
    val,img=capture.read()
    if extract(img) is not None:
        x=x+1
        face=cv2.resize(extract(img),(200,200))
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

        fpath='./sample/'+str(x)+'.jpg'
        cv2.imwrite(fpath,face)

        cv2.putText(face,str(x),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(20,255,90),2)
        cv2.imshow('Cropper',face)
    else:
        pass
    if cv2.waitKey(1)==13 or x==100:
        break
capture.release()
cv2.destroyAllWindows()
print("Collecting Samples Complete")



