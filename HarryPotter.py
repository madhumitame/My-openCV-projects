import cv2
import numpy as np

def hello(x):
    print(' ')

#initialising of the camera
cap = cv2.VideoCapture(0)
bars = cv2.namedWindow("Bars")

cv2.createTrackbar("upper_hue", 'Bars', 110, 180, hello)
cv2.createTrackbar("upper_saturation", 'Bars', 255, 255, hello)
cv2.createTrackbar("upper_value", 'Bars', 255, 255, hello)
cv2.createTrackbar("lower_hue", 'Bars', 68, 180, hello)
cv2.createTrackbar("lower_saturation", 'Bars', 55, 255, hello)
cv2.createTrackbar("lower_value", 'Bars', 54, 255, hello)

# Capturing initial frame
while (True):
    cv2.waitKey(1000)
    ret, init_frame = cap.read()
    # ret will be true if the frame is indeed returned
    if (ret):
        break

while(True):
    ret, frame = cap.read()
    inspect = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #getting hsv values for masking cloak
    upper_hue = cv2.getTrackbarPos("upper_hue", "Bars")
    upper_saturation = cv2.getTrackbarPos("upper_saturation", "Bars")
    upper_value = cv2.getTrackbarPos("upper_value", "Bars")
    lower_hue = cv2.getTrackbarPos("lower_hue", "Bars")
    lower_saturation = cv2.getTrackbarPos("lower_saturation", "Bars")
    lower_value = cv2.getTrackbarPos("lower_value", "Bars")

    #Kernel to be used for dilation i.e. noise reduction
    kernel = np.ones((3, 3), np.uint8)

    # identifying the bedsheet
    upper_hsv = np.array([upper_hue, upper_saturation, upper_value])
    lower_hsv = np.array([lower_hue, lower_saturation, lower_value])

    # making a mask by subracting the bedsheet
    mask = cv2.inRange(inspect, lower_hsv, upper_hsv)
    mask = cv2.medianBlur(mask, 3) # remove impurities
    mask_inv = 255 - mask
    mask = cv2.dilate(mask, kernel, 5) # remove finer impurities

    # adding reference init _frame
    b = frame[:, :, 0]  # splitting b, g, r channels
    g = frame[:, :, 1]
    r = frame[:, :, 2]
    b = cv2.bitwise_and(mask_inv, b) #frame area will remain
    g = cv2.bitwise_and(mask_inv, g)
    r = cv2.bitwise_and(mask_inv, r)
    frame_inv = cv2.merge((b, g, r))

    b = init_frame[:, :, 0]
    g = init_frame[:, :, 1]
    r = init_frame[:, :, 2]
    b = cv2.bitwise_and(b, mask)    # only sheet area will remain
    g = cv2.bitwise_and(g, mask)
    r = cv2.bitwise_and(r, mask)
    blanket_area = cv2.merge((b, g, r))

    final = cv2.bitwise_or(frame_inv, blanket_area)

    cv2.imshow("Invisibility Cloak!", final)
    cv2.imshow("Original", frame)

    if (cv2.waitKey(3) == ord('q')):
        break;

cv2.destroyAllWindows()
cap.release()
