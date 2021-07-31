import cv2
import numpy as np
from collections import deque

# Default function to call trackbar
def setValues(x):
    print("")

# create trackbars
cv2.namedWindow("Colour Detectors")
cv2.createTrackbar("Upper_hue", "Colour Detectors", 153, 180, setValues)
cv2.createTrackbar("Upper_saturation", "Colour Detectors", 255, 255, setValues)
cv2.createTrackbar("Upper_value", "Colour Detectors", 255, 255, setValues)
cv2.createTrackbar("lower_hue", "Colour Detectors", 64, 180, setValues)
cv2.createTrackbar("lower_saturation", "Colour Detectors", 171, 255, setValues)
cv2.createTrackbar("lower_value", "Colour Detectors", 78, 255, setValues)

# Different arrays to handle colour points
bpoints = [deque(maxlen = 1024)]
gpoints = [deque(maxlen = 1024)]
rpoints = [deque(maxlen = 1024)]
ypoints = [deque(maxlen = 1024)]

# Indexes used to mark points of specific colour
b_index = 0
g_index = 0
r_index = 0
y_index = 0

# Kernel for dilation i.e. noise reduction
kernel = np.ones((5, 5), np.uint8)

colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colourIndex = 0

# Canvas
paintWindow = np.zeros((471, 636, 3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), colours[0], -1)
paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), colours[1], -1)
paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), colours[2], -1)
paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), colours[3], -1)

cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Switching on the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # Flipping the frame
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    u_hue = cv2.getTrackbarPos("Upper hue", 'Colour Detectors')
    u_saturation = cv2.getTrackbarPos("Upper Saturation", 'Colour Detectors')
    u_value = cv2.getTrackbarPos("Upper value", 'Colour Detectors')
    l_hue = cv2.getTrackbarPos("Lower hue", 'Colour Detectors')
    l_saturation = cv2.getTrackbarPos("Lower Saturation", 'Colour Detectors')
    l_value = cv2.getTrackbarPos("Lower value", 'Colour Detectors')
    upper_hsv = np.array([u_hue, u_saturation, u_value])
    lower_hsv = np.array([l_hue, l_saturation, l_value])

    # Buttons on the live frame
    frame = cv2.rectangle(frame, (40, 1), (140, 65), (122, 122, 122), -1)
    frame = cv2.rectangle(frame, (160, 1), (255, 65), colours[0], -1)
    frame = cv2.rectangle(frame, (275, 1), (370, 65), colours[1], -1)
    frame = cv2.rectangle(frame, (390, 1), (485, 65), colours[2], -1)
    frame = cv2.rectangle(frame, (505, 1), (600, 65), colours[3], -1)

    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2, cv2.LINE_AA)

    # Identify the bead and remove any noise
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2. dilate(mask, kernel, iterations = 1)

    # Find contour
    cnt, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if len(cnt) > 0:
        # sorting cnts
        cnts = sorted(cnt, key=cv2.contourArea, reverse=True)[0]
        # Get the radius of the enclosing circle around the found contour
        ((x, y), radius) = cv2.minEnclosingCircle(cnts)
        # draw a circle
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        # calculating the center of the detected contour
        M = cv2.moments(cnts)
        center = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))

        # Check if the user wants to click any of the buttons
        if center[1] <= 65:
            if 40 <= center[0] <= 140:  # clear button
                bpoints = [deque(maxlen = 512)]
                gpoints = [deque(maxlen = 512)]
                rpoints = [deque(maxlen = 512)]
                ypoints = [deque(maxlen = 512)]

                b_index = 0
                g_index = 0
                r_index = 0
                y_index = 0

                paintWindow[67:, :, :] = 255

            elif 160 <= center[0] <= 255:
                colourIndex = 0  # Blue
            elif 275 <= center[0] <= 370:
                colourIndex = 1  # Green
            elif 390 <= center[0] <= 485:
                colourIndex = 2  # Red
            elif 505 <= center[0] <= 600:
                colourIndex = 3  # Yellow

        else:
            if colourIndex == 0:
                bpoints[b_index].appendleft(center)
            elif colourIndex == 1:
                gpoints[g_index].appendleft(center)
            elif colourIndex == 2:
                rpoints[r_index].appendleft(center)
            elif colourIndex == 3:
                ypoints[y_index].appendleft(center)

    # Append the next dequeues when nothing is detected, so that there is no lines drawn when the bead is visible again
    else:
        bpoints.append(deque(maxlen = 512))
        b_index += 1
        gpoints.append(deque(maxlen=512))
        g_index += 1
        rpoints.append(deque(maxlen=512))
        r_index += 1
        ypoints.append(deque(maxlen=512))
        y_index += 1

    # draw lines of all colors on the canvas and frame
    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k-1], points[i][j][k], colours[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colours[i], 2)

    # Show all the windows
    cv2.imshow('Track', frame)
    cv2.imshow('Paint', paintWindow)
    cv2.imshow('Mask', mask)

    # Press q to stop the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all the windows
cap.release()
cv2.destroyAllWindows()