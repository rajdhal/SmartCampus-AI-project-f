from image_recognition import crop
import cv2 as cv

file_name = 'test2.JPEG'
img = cv.imread(file_name)
img_cropped = crop(img)

