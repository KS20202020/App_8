import glob
import cv2
import time
from emailing import send_email

video = cv2.VideoCapture(0)
time.sleep(1)

first_frame = None
status_list = []
count = 1

while True:
    status = 0
    check, frame = video.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame_gau = cv2.GaussianBlur(gray_frame,(21,21),0)

    if first_frame is None:
        first_frame = gray_frame_gau

    delta_frame = cv2.absdiff(first_frame, gray_frame_gau)
    thresh_frame = cv2.threshold(delta_frame,70,255,cv2.THRESH_BINARY)[1]
    dil_frame = cv2.dilate(thresh_frame,None, iterations=5)
    cv2.imshow('my_video', dil_frame)

    contours, check = cv2.findContours(dil_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 5000:
            continue
        x, y, h, w = cv2.boundingRect(contour)
        rectangle = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        if rectangle.any():
            status = 1
            cv2.imwrite(f'images/{count}.png', frame)
            count = count + 1
            image_list = glob.glob('images/*.png')
            index = int(len(image_list)) / 2
            index = int(index)
            image_with_object = image_list[index]


    status_list.append(status)
    status_list = status_list[-2:]
    if status_list[0] == 1 and status_list[1] == 0:
        send_email(image_with_object)
    print(status_list)

    cv2.imshow('Video',frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
