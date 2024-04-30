import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import glob
import os
import os.path

#this function removes the margins of the image so that the letter is not cut off and no noise in the image is present
def removeMargins(img):
    th, threshed = cv.threshold(img, 245, 255, cv.THRESH_BINARY_INV)
    ## (2) Morph-op to remove noise
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11,11))
    morphed = cv.morphologyEx(threshed, cv.MORPH_CLOSE, kernel)
    ## (3) Find the max-area contour
    cnts = cv.findContours(morphed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv.contourArea)[-1]
    ## (4) Crop and save it
    x,y,w,h = cv.boundingRect(cnt)
    dst = img[y:y+h, x:x+w]
    return dst
def binary_otsus(image, filter:int=1):
    """Binarize an image 0's and 255's using Otsu's Binarization"""

    if len(image.shape) == 3:
        gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray_img = image

    # Otsus Binarization
    if filter != 0:
        blur = cv.GaussianBlur(gray_img, (3,3), 0)
        binary_img = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
    else:
        binary_img = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
    
    # Morphological Opening
    # kernel = np.ones((3,3),np.uint8)
    # clean_img = cv.morphologyEx(binary_img, cv.MORPH_OPEN, kernel)

    return binary_img
def whiteBlackRatio(img):
    h = img.shape[0]
    w = img.shape[1]
    blackCount=1
    whiteCount=0
    for y in range(0,h):
        for x in range (0,w):
            if (img[y,x]==0):
                blackCount+=1
            else:
                whiteCount+=1
    return whiteCount/blackCount
def horizontalTransitions(img):
    transitions = np.sum(np.abs(np.diff(img, axis=1)) > 0, axis=1)
    if len(transitions) == 0:
        return 0
    return np.max(transitions)
def verticalTransitions(img):
    transitions = np.sum(np.abs(np.diff(img, axis=0)) > 0, axis=0)
    if len(transitions) == 0:
        return 0
    return np.max(transitions)
    
def blackPixelsCount(img):
    blackCount=1 #initialized at 1 to avoid division by zero when we calculate the ratios
    h = img.shape[0]
    w = img.shape[1]
    for y in range(0,h):
        for x in range (0,w):
            if (img[y,x]==0):
                blackCount+=1
            
    return blackCount
def histogramAndCenterOfMass(img):
    h = img.shape[0]
    w = img.shape[1]
    histogram=[]
    sumX=0
    sumY=0
    num=0
    for x in range(0,w):
        localHist=0
        for y in range (0,h):
            if(img[y,x]==0):
                sumX+=x
                sumY+=y
                num+=1
                localHist+=1
        histogram.append(localHist)
      
    return sumX/num , sumY/num, histogram
def getFeatures(image):
    x,y=image.shape
    featuresList=[]
    # first feature: height/width ratio
    featuresList.append(y/x)
    #Second feature: white/black ratio
    featuresList.append(whiteBlackRatio(image))
    #Third feature: horizontal transitions   
    featuresList.append(horizontalTransitions(image))
    #Forth feature : vertical transitions
    featuresList.append(verticalTransitions(image))
    #Fifth feature : splitting the image into 4 images
    topLeft=image[0:y//2,0:x//2]
    topRight=image[0:y//2,x//2:x]
    bottomeLeft=image[y//2:y,0:x//2]
    bottomRight=image[y//2:y,x//2:x]

    #get white to black ratio in each quarter
    featuresList.append(whiteBlackRatio(topLeft))
    featuresList.append(whiteBlackRatio(topRight))
    featuresList.append(whiteBlackRatio(bottomeLeft))
    featuresList.append(whiteBlackRatio(bottomRight))
    #the next 6 features are:
    #• Black Pixels in Region 1/ Black Pixels in Region 2.
    #• Black Pixels in Region 3/ Black Pixels in Region 4.
    #• Black Pixels in Region 1/ Black Pixels in Region 3.
    #• Black Pixels in Region 2/ Black Pixels in Region 4.
    #• Black Pixels in Region 1/ Black Pixels in Region 4
    #• Black Pixels in Region 2/ Black Pixels in Region 3.
    topLeftCount=blackPixelsCount(topLeft)
    topRightCount=blackPixelsCount(topRight)
    bottomLeftCount=blackPixelsCount(bottomeLeft)
    bottomRightCount=blackPixelsCount(bottomRight)

    featuresList.append(topLeftCount/topRightCount)
    featuresList.append(bottomLeftCount/bottomRightCount)
    featuresList.append(topLeftCount/bottomLeftCount)
    featuresList.append(topRightCount/bottomRightCount)
    featuresList.append(topLeftCount/bottomRightCount)
    featuresList.append(topRightCount/bottomLeftCount)
    #get center of mass and horizontal histogram
    xCenter, yCenter,xHistogram =histogramAndCenterOfMass(image)
    featuresList.append(xCenter)
    featuresList.append(yCenter)
    #featuresList.extend(xHistogram)
    print(len(featuresList))
    return featuresList
    

# img = cv.imread("char4.png", cv.IMREAD_GRAYSCALE)
# cv.imshow("Image", img)
# cv.waitKey(0)
# cv.destroyAllWindows()
# cropped = removeMargins(img)
# #show image with margins removed
# cv.imshow("Image", cropped)
# cv.waitKey(0)
# cv.destroyAllWindows()
def trainAndClassify(data,classes):
    
    X_train, X_test, y_train, y_test = train_test_split(data, classes, test_size = 0.20)
    svclassifier = SVC(kernel='rbf', gamma =0.005 , C =1000)
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
        if os.path.isdir(os.path.join(a_dir, name))]
def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles 
def main():
    data=np.array([])
    classes=np.array([])
    directory=r'F:\LockD\CMP2025\Third_Year\Second_Term\Neural_Networks\Project\training_set\IBM'
    chars=get_immediate_subdirectories(directory)
    count=0
    numOfFeatures=16
    charPositions=['Beginning','End','Isolated','Middle']
    for char in chars:
        for position in charPositions:
            if(os.path.isdir(directory+'/'+char+'/'+position)==True):
                listOfFiles = getListOfFiles(directory+'/'+char+'/'+position)
                for filename in listOfFiles:
                    img = cv.imread(filename)
                    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    cropped = removeMargins(gray_img)
                    binary_img = binary_otsus(cropped, 0)
                    features=getFeatures(binary_img)
                    data= np.append(data,features)
                    classes=np.append(classes,char+position)
                    count+=1
    
    data=np.reshape(data,(count,numOfFeatures))
    trainAndClassify(data,classes)
