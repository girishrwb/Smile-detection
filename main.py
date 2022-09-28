import cv2

#get the ml data
trained_face_data = cv2.CascadeClassifier('Default_frontal_face.xml')
Smile_detector = cv2.CascadeClassifier('Smile_detection.xml')

#get the webcam feed
webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()

    if not successful_frame_read: break

    grey_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_co = trained_face_data.detectMultiScale(grey_img)


    for(x,y,w,h) in face_co:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        #the_face = (x,y,(x+w), (y+h))
        the_face = frame[y:y+h, x:x+w]

        face_grey = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        smile_co = Smile_detector.detectMultiScale(face_grey, scaleFactor=1.7, minNeighbors=30)

        #or(x_, y_, w_, h_) in smile_co:
            #cv2.rectangle(the_face, (x_, y_), (x_+h_, y_+h_), (255, 255, 0), 2)
        if len(smile_co) > 0:
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=2, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,255,255) )
    cv2.imshow('Smile Detector', frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break
webcam.release()
cv2.destroyAllWindows()


print("works fine!")