import numpy as np
import cv2

img = cv2.imread("Competition Details.png")
img1 = cv2.imread("Sample1.png")

print(img.shape) # return a tuple of rows columns and channels
print(img1.shape)
print(img.size) # Total pixels
print(img.dtype)
b, g, r = cv2.split(img)
img2 = cv2.merge((b, g, r))

img = cv2.resize(img, (1000, 1000))
img1 = cv2.resize(img1, (1000, 1000))

dst = cv2.add(img, img1)
dst2 = cv2.addWeighted(img, .9, img1, .1, 0)

cv2.imshow('image', dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#bitwise operation
