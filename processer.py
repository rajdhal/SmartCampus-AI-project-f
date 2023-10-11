import cv2 as cv


def process(img):
    # Convert to grayscale
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    cv.imwrite('result3.jpg', img)  # save image
    return img