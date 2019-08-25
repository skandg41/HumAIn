from lib import Tools

import cv2
import numpy as np
import math

img = cv2.imread("Test Images/1.jpeg")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hue, saturation, value = cv2.split(hsv)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

topHat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)
blackHat = cv2.morphologyEx(value, cv2.MORPH_BLACKHAT, kernel)


add = cv2.add(value, topHat)
subtract = cv2.subtract(add, blackHat)

blur = cv2.GaussianBlur(subtract, (5, 5), 0)

thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)

cv2MajorVersion = cv2.__version__.split(".")[0]

if int(cv2MajorVersion) >= 4:
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
else:
    imageContours, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


height, width = thresh.shape

imageContours = np.zeros((height, width, 3), dtype=np.uint8)

possibleChars = []
countOfPossibleChars = 0

for i in range(0, len(contours)):
    cv2.drawContours(imageContours, contours, i, (255, 255, 255))
    possibleChar = Tools.ifChar(contours[i])
    if Tools.checkIfChar(possibleChar) is True:
        countOfPossibleChars = countOfPossibleChars + 1
        possibleChars.append(possibleChar)


imageContours = np.zeros((height, width, 3), np.uint8)

ctrs = []

for char in possibleChars:
    ctrs.append(char.contour)

cv2.drawContours(imageContours, ctrs, -1, (255, 255, 255))


plates_list = []
listOfListsOfMatchingChars = []

for possibleC in possibleChars:

    def matchingChars(possibleC, possibleChars):
        listOfMatchingChars = []

        for possibleMatchingChar in possibleChars:
            if possibleMatchingChar == possibleC:
                continue

            distanceBetweenChars = Tools.distanceBetweenChars(possibleC, possibleMatchingChar)

            angleBetweenChars = Tools.angleBetweenChars(possibleC, possibleMatchingChar)

            changeInArea = float(abs(possibleMatchingChar.boundingRectArea - possibleC.boundingRectArea)) / float(
                possibleC.boundingRectArea)

            changeInWidth = float(abs(possibleMatchingChar.boundingRectWidth - possibleC.boundingRectWidth)) / float(
                possibleC.boundingRectWidth)

            changeInHeight = float(abs(possibleMatchingChar.boundingRectHeight - possibleC.boundingRectHeight)) / float(
                possibleC.boundingRectHeight)

            if distanceBetweenChars < (possibleC.diagonalSize * 5) and \
                    angleBetweenChars < 12.0 and \
                    changeInArea < 0.5 and \
                    changeInWidth < 0.8 and \
                    changeInHeight < 0.2:
                listOfMatchingChars.append(possibleMatchingChar)

        return listOfMatchingChars


    listOfMatchingChars = matchingChars(possibleC, possibleChars)

    listOfMatchingChars.append(possibleC)

    if len(listOfMatchingChars) < 3:
        continue

    listOfListsOfMatchingChars.append(listOfMatchingChars)

    listOfPossibleCharsWithCurrentMatchesRemoved = list(set(possibleChars) - set(listOfMatchingChars))

    recursiveListOfListsOfMatchingChars = []

    for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:
        listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)

    break

imageContours = np.zeros((height, width, 3), np.uint8)

for listOfMatchingChars in listOfListsOfMatchingChars:
    contoursColor = (255, 0, 255)

    contours = []

    for matchingChar in listOfMatchingChars:
        contours.append(matchingChar.contour)

    cv2.drawContours(imageContours, contours, -1, contoursColor)


for listOfMatchingChars in listOfListsOfMatchingChars:
    possiblePlate = Tools.PossiblePlate()

    listOfMatchingChars.sort(key=lambda matchingChar: matchingChar.centerX)

    plateCenterX = (listOfMatchingChars[0].centerX + listOfMatchingChars[len(listOfMatchingChars) - 1].centerX) / 2.0
    plateCenterY = (listOfMatchingChars[0].centerY + listOfMatchingChars[len(listOfMatchingChars) - 1].centerY) / 2.0

    plateCenter = plateCenterX, plateCenterY

    plateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].boundingRectX + listOfMatchingChars[
        len(listOfMatchingChars) - 1].boundingRectWidth - listOfMatchingChars[0].boundingRectX) * 1.3)

    totalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        totalOfCharHeights = totalOfCharHeights + matchingChar.boundingRectHeight

    averageCharHeight = totalOfCharHeights / len(listOfMatchingChars)

    plateHeight = int(averageCharHeight * 1.5)

    opposite = listOfMatchingChars[len(listOfMatchingChars) - 1].centerY - listOfMatchingChars[0].centerY

    hypotenuse = Tools.distanceBetweenChars(listOfMatchingChars[0],
                                                listOfMatchingChars[len(listOfMatchingChars) - 1])
    correctionAngleInRad = math.asin(opposite / hypotenuse)
    correctionAngleInDeg = correctionAngleInRad * (180.0 / math.pi)

    possiblePlate.rrLocationOfPlateInScene = (tuple(plateCenter), (plateWidth, plateHeight), correctionAngleInDeg)

    rotationMatrix = cv2.getRotationMatrix2D(tuple(plateCenter), correctionAngleInDeg, 1.0)

    height, width, numChannels = img.shape

    imgRotated = cv2.warpAffine(img, rotationMatrix, (width, height))

    imgCropped = cv2.getRectSubPix(imgRotated, (plateWidth, plateHeight), tuple(plateCenter))

    possiblePlate.Plate = imgCropped

# Applying OCR

import pytesseract 
# Append address for Tesseract exe file installed in your system here
pytesseract.pytesseract.tesseract_cmd="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def adaptiveT(plate):    
    img = plate
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    threshGauss = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 27)
    cv2.imwrite("processed.png", threshGauss)
    cv2.waitKey(0)


def reSize():
    img = cv2.imread("processed.png")
    ratio = 200.0 / img.shape[1]
    dim = (200, int(img.shape[0] * ratio))
    resizedCubic = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("resized.png", resizedCubic)



def aBorder():    
    img = cv2.imread("resized.png")
    bordersize = 10
    border = cv2.copyMakeBorder(img, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize,
                                borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
    cv2.imwrite("borders.png", border)



def cleanOCR():
    detectedOCR = []
    img = cv2.imread("borders.png")
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    gray = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se)
    config = '-l eng --oem 1 --psm 3'
    text = pytesseract.image_to_string(gray, config=config)
    validChars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                  'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    cleanText = []
    for char in text:
        if char in validChars:
            cleanText.append(char)

    plate = ''.join(cleanText)
    detectedOCR.append(plate)
    return detectedOCR



adaptiveT(imgCropped)
reSize()
aBorder()
platesL = cleanOCR()

print(platesL[0])
cv2.waitKey(0)
