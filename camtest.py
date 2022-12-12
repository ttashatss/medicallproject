import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    print('looping')
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
    
#   if cv2.waitKey(1) and 0xFF == ord('q'):
#		break

cap.release()
cv2.destroyAllWindows()