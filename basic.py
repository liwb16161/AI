import numpy as np
import cv2 as cv
original = cv.imread('../data/forest.jpg')
cv.imshow('Original', original)
blue = np.zeros_like(original)
blue[..., 0] = original[..., 0]
cv.imshow('Blue', blue)
green = np.zeros_like(original)
green[..., 1] = original[..., 1]
cv.imshow('Green', green)
red = np.zeros_like(original)
red[..., 2] = original[..., 2]
cv.imshow('Red', red)
h, w = original.shape[:2]
l, t = int(w / 4), int(h / 4)
r, b = int(w * 3 / 4), int(h * 3 / 4)
cropped = original[t:b, l:r]
cv.imshow('Cropped', cropped)
#scaled = cv.resize(original,
#	(int(w / 2), int(h / 2)),
#	interpolation=cv.INTER_LINEAR)
scaled = cv.resize(original, None,
	fx=2, fy=2,
	interpolation=cv.INTER_LINEAR)
cv.imshow('Scaled', scaled)
cv.waitKey();
cv.imwrite('./cropped.bmp', cropped)
cv.imwrite('./cropped.png', cropped)
cv.imwrite('./cropped.jpg', cropped)