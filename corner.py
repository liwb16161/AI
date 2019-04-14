import cv2 as cv
original = cv.imread('../data/box.png',
	cv.IMREAD_GRAYSCALE)
cv.imshow('Original', original)
corners = cv.cornerHarris(original, 7, 5, 0.04)
corners = cv.dilate(corners, None)
mixture = original.copy()
mixture[corners > corners.max() * 0.01] = 255
cv.imshow('Mixture', mixture)
cv.waitKey()
