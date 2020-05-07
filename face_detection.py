import os
import cv2

os.chdir("C:/Users/abdul/OneDrive/Pictures/New folder")

cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

image = cv2.imread("two_people.jpg")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img = cascade.detectMultiScale(image_gray,scaleFactor=1.1,minNeighbors=10)
for x, y, w, h in img:
    rec = cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)

final = cv2.resize(rec , (int(rec.shape[1]/1.5), int(rec.shape[0]/1.5)))
cv2.imshow("image_gray", final)
cv2.waitKey(0)
cv2.destroyAllWindows()