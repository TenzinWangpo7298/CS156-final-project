import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import glob
import math

# here we will get all the jpg image from the test image and store it in testImage array
testImage = [path for path in glob.glob("test_images/*.jpg")]

"""
imageShow function will take the list of images and show them in three column
using matplotlib, and we will use this function for all the test
"""


def imageShow(images, cmap=None):
    column = 2
    row = (len(images) + 1) // column
    plt.figure(figsize=(6, 9))
    for num, image in enumerate(images):
        plt.subplot(row, column, num + 1)
        cmap = 'gray' if len(image.shape) == 2 else cmap
        plt.imshow(image, cmap)
        plt.xticks([])
        plt.yticks([])
    plt.show()


"""
This function is to convert the RGB image to grayscale image using OpenCV
"""


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


"""
this function will convert the RGB Image to HLS image using OpenCV
"""


def convertImageToHLS(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)


"""
this function will select the white and yellow pixel value only, if the pixel value is 
not in the range of yellow and white threshold then it will make it 0, or black
because usually the highway road lane color is either yellow or white and all other color are not useful
"""


def selectWhiteAndYellow(image):
    # lower and upper threshold for white pixel
    lowerWhite = np.uint8([190, 190, 190])
    upperWhite = np.uint8([255, 255, 255])
    # lower and upper threshold for yellow pixel
    lowerYellow = np.uint8([190, 190, 0])
    upperYellow = np.uint8([255, 255, 255])
    # extract the pixel which is in white threshold and all other make it black
    maskWhite = cv2.inRange(image, lowerWhite, upperWhite)
    # same here except it is yellow threshold, all other black
    maskYellow = cv2.inRange(image, lowerYellow, upperYellow)
    # here we will take both white and yellow threshold and if the pixel is either in
    # white and yellow threshold, then we choose it else make the pixel black or ignore
    orMask = cv2.bitwise_or(maskWhite, maskYellow)
    # then we apply this mask to the input image, so basically if the pixel in orMask image is white
    # then it will retain the pixel value, else make it 0 or black
    # or we can define as if both image and ormask has value,
    # then retain the pixel value else make 0 (and definition)
    final_mask = cv2.bitwise_and(image, image, mask=orMask)
    return final_mask


"""
this function is normalizing the pixel value with surrounding pixel value, 
or in other terms, it is blurring ro smoothing the image by convolving the kernel size 
square matrix over the image and applying the Gaussian algorithm. 
"""


def gaussainBlur(image, kernelSize=13):
    return cv2.GaussianBlur(image, (kernelSize, kernelSize), 0)


"""
this function is using the Canny edge detection algorithm from OpenCV library to detect the edge, 
"""


def cannyEdgeDetection(image, lowerThreshol=50, upperThreshold=150):
    return cv2.Canny(image, lowerThreshol, upperThreshold)


"""
ROI is Region Of Interest, where we basically make unwanted information in image to 0 or black, 
for example, the upper half part of image is useless where most of them are sky
so our region of interest is lower half part of image, so we make upper part black, and retain
the lower part. 
"""


def ROI(image):
    # first we will create a same size of image with all black pixel or black image
    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        ROIMask = (255,) * image.shape[2]
    else:
        ROIMask = 255

    # here are the vertices for our ROI which will be white and it will be draw over black mask
    row, col = image.shape[:2]
    left_lower = [col * 0.05, row * 1.0]
    left_upper = [col * 0.4, row * 0.6]
    right_lower = [col * 0.95, row * 1.0]
    right_upper = [col * 0.6, row * 0.6]
    vertices = np.array([[left_lower, left_upper, right_upper, right_lower]], dtype=np.int32)

    # here it will draw the above given vertices shape on black mask that we create at beginning
    cv2.fillPoly(mask, vertices, ROIMask)

    ROI_image = cv2.bitwise_and(image, mask)
    return ROI_image


def houghLineTransformation(image):
    return cv2.HoughLinesP(image, rho=1, theta=np.pi / 180, threshold=20, minLineLength=20, maxLineGap=300)


def pipeline(image):
    yellowNWhite = selectWhiteAndYellow(image)
    grayscaleImage = grayscale(yellowNWhite)
    blurImage = gaussainBlur(grayscaleImage)
    detectEdges = cannyEdgeDetection(blurImage)
    ROIImage = ROI(detectEdges)
    lineDetectPixel = houghLineTransformation(ROIImage)
    for line in lineDetectPixel:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color=[255, 0, 0], thickness=2)
    return image


test_images = [plt.imread(image_path) for image_path in testImage]
grayscaleImage = [grayscale(image) for image in test_images]
hlsImages = [pipeline(image) for image in test_images]
imageShow(hlsImages)
