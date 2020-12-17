import matplotlib.pyplot as plt
import cv2
import numpy as np


def imageShow(images, cmap=None):
    """
    imageShow function will take the list of images and show them in three column
    using matplotlib, and we will use this function for all the test
    """
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


def grayscale(image):
    """
    This function is to convert the RGB image to grayscale image using OpenCV
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def convertImageToHLS(image):
    """
    this function will convert the RGB Image to HLS image using OpenCV
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)


def selectWhiteAndYellow(image):
    """
    this function will select the white and yellow pixel value only, if the pixel value is
    not in the range of yellow and white threshold then it will make it 0, or black
    because usually the highway road lane color is either yellow or white and all other color are not useful
    """
    # lower and upper threshold for white pixel
    lowerWhite = np.uint8([0, 200, 0])
    upperWhite = np.uint8([255, 255, 255])
    # lower and upper threshold for yellow pixel
    lowerYellow = np.uint8([10, 0, 100])
    upperYellow = np.uint8([40, 255, 255])
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


def gaussainBlur(image, kernelSize=15):
    """
    this function is normalizing the pixel value with surrounding pixel value,
    or in other terms, it is blurring ro smoothing the image by convolving the kernel size
    square matrix over the image and applying the Gaussian algorithm.
    """
    return cv2.GaussianBlur(image, (kernelSize, kernelSize), 0)


def cannyEdgeDetection(image, lowerThreshold=50, upperThreshold=150):
    """
    this function is using the Canny edge detection algorithm from OpenCV library to detect the edge,
    """
    return cv2.Canny(image, lowerThreshold, upperThreshold)


def ROI(image):
    """
    ROI is Region Of Interest, where we basically make unwanted information in image to 0 or black,
    for example, the upper half part of image is useless where most of them are sky
    so our region of interest is lower half part of image, so we make upper part black, and retain
    the lower part.
    """
    # first we will create a same size of image with all black pixel or black image
    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        ROIMask = (255,) * image.shape[2]
    else:
        ROIMask = 255

    # here are the vertices for our ROI which will be white and it will be draw over black mask
    row, col = image.shape[:2]
    left_lower = [col * 0.1, row * 0.95]
    left_upper = [col * 0.4, row * 0.6]
    right_lower = [col * 0.9, row * 0.95]
    right_upper = [col * 0.6, row * 0.6]
    vertices = np.array([[left_lower, left_upper, right_upper, right_lower]], dtype=np.int32)

    # here it will draw the above given vertices shape on black mask that we create at beginning
    cv2.fillPoly(mask, vertices, ROIMask)

    ROI_image = cv2.bitwise_and(image, mask)
    return ROI_image


def houghLineTransformation(image):
    """
    this is opencv function which it use the polar coordinate and it is
    the Probabilistic Hough Line Transformation. it output the extremes of the detected lines (x1,y1,x2,y2)
    based on the parameter given below
    :param image:
    :return [(x1,y1,x2,y2)...]:
    """
    return cv2.HoughLinesP(image, rho=1, theta=np.pi / 180, threshold=20, minLineLength=20, maxLineGap=300)


def slope(x1, y1, x2, y2):
    """
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return slope:
    """
    return (y2 - y1) / ((x2 - x1) * 1.0)


def yIntercept(x, y, slope):
    """
    for y-intercept, both y1 and y2 or x2 or x1 will have same y-intercept since it is same line
    so here we will take y1 and x1

    :param x coordinate:
    :param y coordinate:
    :param slope:
    :return y-intercept :
    """
    return y - (slope * x)


def bestLine(linesFromHoughLineP):
    """
    Here input is from the output of hough Line transformation

    The idea here is that hough line output many line that it think is line,
    so we need to pick the best two single line which will represent the left lane
    and right lane. Here we will use dictionary, where key is y-intercept and value is list
    of slope and y-intercept

    To do that we will loop though the output of hough line and separate them by slope value
    Slope = (y2 - y1) / (x2 - x1)
    since cordinate of the image is reverse as 0,0 is at the upper left side, so y coordinate
    is also reverse so left lane (in normal coordinate, it is slope up) but here left lane is
    negative slope and right lane is positive slope

    as we separate the slope, we will also calculate the y-intercept, and it will be stored in
    different array, later, y-intercept will decide which is the best line by taking the median.

    y = mx + c         m = slope,  x = input, c = y-intercept
    c = y - mx         y-intercept formula

    once we get the median of y-intercept for both lanes, then we find the slope and y-intercept for this line
    in dictionary using y-intercept

    :param linesFromHoughLineP:
    :return [[left Lane line slope and y-intercept], [right lane line slope and y-intercept]]:
    """
    left_lane = dict()
    right_lane = dict()
    left_intercept = []
    right_intercept = []

    for line in linesFromHoughLineP:
        x1, y1, x2, y2 = line[0][0], line[0][1], line[0][2], line[0][3]
        if x1 == x2:
            # EDGE CASE: if x1 and x2 are same, then it is point
            continue
        m = slope(x1, y1, x2, y2)
        c = yIntercept(x1, y1, m)
        if m > 0:
            # it is right lane as I have describe above
            right_lane[c] = [m, c]
            right_intercept.append(c)
        elif m < 0:
            left_lane[c] = [m, c]
            left_intercept.append(c)
        else:
            # if it 0, then ignore it because lane can't be straight line with no slope
            continue
            
    # now we have separated the both left lane and right lane
    medianYinterceptLeftLane = left_intercept[len(left_intercept) // 2]
    medianYinterceptRightLane = right_intercept[len(right_intercept) // 2]
    # we find the best lane coordinate in dictionary using median y-intercept as key
    bestLeftLaneLine = left_lane[medianYinterceptLeftLane]
    bestRightLaneLine = right_lane[medianYinterceptRightLane]
    bestLineForEachLane = [bestLeftLaneLine, bestRightLaneLine]
    return bestLineForEachLane


def extrapolateLine(image, lines):
    """
    Here we are extrapolating the line where hough line Transformation detect the line only
    the pixel detect by Canny edge detection, so it will be short line. In this function we are
    extending the line by using the slope and intercept we get from bestLine.

    We will calculate the x1,y1,x2,y2 from slope and intercept using y = mx + c (line equation)
    :param image:
    :param lines:
    :return [[leftLane line coordinate], [right lane line coordinate]]:
    """
    slope_left = lines[0][0]
    yIntercept_left = lines[0][1]
    slope_right = lines[1][0]
    yIntercept_right = lines[1][1]

    y1 = image.shape[0]
    y2 = int(image.shape[0] // 1.6)

    x1_left = int((y1 - yIntercept_left) / slope_left)
    x2_left = int((y2 - yIntercept_left) / slope_left)

    x1_right = int((y1 - yIntercept_right) / slope_right)
    x2_right = int((y2 - yIntercept_right) / slope_right)

    return [[x1_left, y1, x2_left, y2], [x1_right, y1, x2_right, y2]]


def pipeline(image):
    roiImages = ROI(image)
    hlsImage = convertImageToHLS(roiImages)
    whiteYellowLine = selectWhiteAndYellow(hlsImage)
    grayscaleImage = grayscale(whiteYellowLine)
    smoothImage = gaussainBlur(grayscaleImage)
    edgeDetect = cannyEdgeDetection(smoothImage)
    probabilisticLineDetect = houghLineTransformation(edgeDetect)
    bestLines = bestLine(probabilisticLineDetect)
    extrapolateLines = extrapolateLine(image, bestLines)
    # now lets draw the line from output extrapolateLines on image
    for line in extrapolateLines:
        x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
        cv2.line(image, (x1, y1), (x2, y2), color=[0, 255, 0], thickness=15)
    return image

