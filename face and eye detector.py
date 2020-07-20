import cv2
import time

face_features=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyes_features=cv2.CascadeClassifier("harrcascase_eye.xml")

video=cv2.VideoCapture(0)

#initial frames
a=0

while True:
    a=a+1
    check,frame=video.read()
    print(check)
    print(frame)

    #if we want to convert our frame into gray frames
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


    #we usually use gray_frame for determining coordinates
    face_coordinates=face_features.detectMultiScale(gray_frame,scaleFactor=1.05,
    minNeighbors=5)

    for x,y,w,h in face_coordinates:
        frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        eyes_color=frame[y:y+h,x:x+w]
        eyes_gray=gray_frame[y:y+h,x:x+w]

        #we usually use gray_eyes for determining coordinates of eyes
        eyes_coordinates=eyes_features.detectMultiScale(eyes_gray)
        for (ex,ey,ew,eh) in eyes_coordinates:
            eyes_color=cv2.rectangle(eyes_color,(ex,ey),(ex+ew,ey+eh),
            (255,0,0),2)


    cv2.imshow("capturing",frame)

    key=cv2.waitKey(1)

    if key==ord('q'):
        break

print(a)

video.release()

cv2.destroyAllWindows()
