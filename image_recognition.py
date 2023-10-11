from ultralytics import YOLO
import cv2 as cv
import numpy as np


def crop(image):
    model = YOLO('runs/detect/train/weights/best.pt')  # load a custom model

    result = model(image)  # predict on an image

    # Best Bounding Box
    bestBox = None
    bestConf = 0

    # Iterate over each box to find box with highest confidence rating
    for box in result[0].boxes:
        if box.conf.cpu().numpy()[0] > bestConf:
            bestConf = box.conf.item()
            bestBox = box

    # IF BESTBOX is not NONE get COORDS
    if bestBox is not None:
        x1, y1, x2, y2 = map(int, bestBox.xyxy.cpu().numpy()[0])

    im_cropped = image[y1:y2, x1:x2]
    gray = cv.cvtColor(im_cropped, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

    filename = 'result4.jpg'
    cv.imwrite(filename,thresh)  # save image
    
    return thresh
    