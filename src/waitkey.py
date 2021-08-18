import cv2
import time

img = cv2.imread("minion.jpg")
cv2.imshow("Flowers",img)
initial_time = time.time()
cv2.waitKey(3000)
final_time = time.time()
print("Window is closed after",(final_time-initial_time))