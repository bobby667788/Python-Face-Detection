__author__ = 'Bobby'
import cv2
import numpy as np
from os import listdir
from os.path import isfile,join
import winsound
fclass=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

tdata,label=[],[]
fpath='./sample/'
files=[f for f in listdir(fpath) if isfile(join(fpath,f))]

for i,file in enumerate(files):
    imgpath=fpath+files[i]
    imgs=cv2.imread(imgpath,cv2.IMREAD_GRAYSCALE)
    tdata.append(np.asarray(imgs,dtype=np.uint8))
    label.append(i)
label=np.asarray(label,dtype=np.int32)
mod=cv2.face.LBPHFaceRecognizer_create()
mod.train(np.asarray(tdata),np.asarray(label))

def detect(img,size=0.5):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=fclass.detectMultiScale(gray,1.3,5)
    if faces is ():
        return img,[]
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        shape=img[y:y+h,x:x+w]
        shape=cv2.resize(shape,(400,600))
    return img,shape

cap=cv2.VideoCapture(0)

while True:
    rte,img=cap.read()
    image,face=detect(img)

    try:
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        result=mod.predict(face)

        if result[1]<500:
            con=int(100*(1-(result[1])/400))
            dispstring=str(con)

        cv2.putText(image,dispstring,(100,120),cv2.FONT_HERSHEY_COMPLEX,1,(255,100,120),2)
        if con>70:
            cv2.putText(image,"Match Found",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow('Cropped',image)

        else:
            cv2.putText(image,"No Match",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            cv2.imshow('Cropped',image)
    except:
            winsound.Beep(1000,1000)
            cv2.putText(image,"No Face Found",(220,120),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            cv2.putText(image,"Locked",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            cv2.imshow('Cropped',image)
            pass

    if cv2.waitKey(1)==13:
        break

cap.release()
cv2.destroyAllWindows()





