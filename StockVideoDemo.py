import numpy as np
import cv2

cap = cv2.VideoCapture("highway.mp4")
#reading first frame
_, first_frame=cap.read()
#reading first frame grayscaled
first_gray=cv2.cvtColor(first_frame,cv2.COLOR_BGR2GRAY)
#first frame for gaussian blur
first_gauss=cv2.GaussianBlur(first_gray, (5, 5), 0)
while True:
    _, frame = cap.read()
    #grayscaling rest of the frames
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #frames with gaussian blur
    gauss_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    #Difference between first frame and the rest in grayscale
    gray_diff=cv2.absdiff(first_gray,gray_frame)
    #Difference between first frame and the rest in color
    difference=cv2.absdiff(first_frame,frame)
    #setting threshold for how clear we want the image to be (Values 0(black)-255(white))
    _, thresh_gray_diff=cv2.threshold(gray_diff,25,255,cv2.THRESH_BINARY)
    #difference for gaussian blur
    gaus_diff=cv2.absdiff(first_gauss,gauss_frame)
    #displaying
    cv2.imshow("First frame",first_frame)
    cv2.imshow("Frame", frame)
    cv2.imshow("Difference",difference)
    cv2.imshow("Grayscaled Difference",gray_diff)
    cv2.imshow("Threshold Applied Grayscaled Difference",thresh_gray_diff)
    cv2.imshow("Gaussian Blur",gaus_diff)
    key = cv2.waitKey(30)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
