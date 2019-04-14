import cv2 as cv
original = cv.imread('../data/chair.jpg',
	cv.IMREAD_GRAYSCALE)
cv.imshow('Original', original)
canny = cv.Canny(original, 50, 240)
cv.imshow('Canny', canny)
cv.waitKey()