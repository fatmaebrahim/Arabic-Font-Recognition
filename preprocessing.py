# Text alignment
# Text size/weight
# Image blur
# Brightness variation
# Text color
# Text rotation
# Salt and pepper noise
import numpy as np
import cv2
import csv
from skimage.feature import canny
from skimage.filters import sobel
import math
from skimage.filters import threshold_otsu
import cv2 as cv

def salt_paper(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised_image = cv2.medianBlur(image, 3)
    return denoised_image


# def text_binary(image):
#     # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # np.set_printoptions(threshold=np.inf)
#     # image_canny = cv2.Canny(grayscale_image,20,120)
#     # return image_canny
#     pixels = image.reshape(-1, 1).astype(np.float32)
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
#     k = 2
#     _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
#     labels = labels.reshape(image.shape)
#     counts = np.bincount(labels.flatten())
#     foreground_cluster = np.argmax(counts)
#     binary_image= np.where(labels == foreground_cluster, 0, 255).astype(np.uint8)
     
#      # Preprocess the given image img.
#     #-----------------------------------------------------------------------------------------------
#     # Convert the image to grayscale
#     #-----------------------------------------------------------------------------------------------
#     #Convert the grayscale image to a binary image. Apply a threshold using Otsu's method on the blurred image.
#     # get the threshold of the image using Otsu's method
#     return binary_image
def text_binary(image, filter:int=1):
    """Binarize an image 0's and 255's using Otsu's Binarization"""

    if len(image.shape) == 3:
        gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray_img = image

    # Otsus Binarization
    if filter != 0:
        blur = cv.GaussianBlur(gray_img, (5,5), 0)
        binary_img = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
    else:
        binary_img = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
    # sum all count of white pixels in the image

    white_count = cv.countNonZero(binary_img)
    #sum all count of black pixels in the image
    black_count = len(binary_img) * len(binary_img[0]) - white_count
    if(white_count > black_count):
        binary_img = 255 - binary_img
    blur = cv.GaussianBlur(binary_img, (7,7), 0)

    # Morphological Opening
    # kernel = np.ones((3,3),np.uint8)
    # clean_img = cv.morphologyEx(binary_img, cv.MORPH_OPEN, kernel)

    return binary_img

def adaptive_line_segmentation(preprocessed_img, segment_size=100):
    histogram = np.sum(preprocessed_img, axis=1)
    n_segments = len(histogram) // segment_size
    thresholds = []
    
    # Compute local thresholds
    for i in range(n_segments + 1):
        start = i * segment_size
        end = min((i + 1) * segment_size, len(histogram))
        local_hist = histogram[start:end]
        if len(local_hist) > 0:
            local_min = np.min(local_hist)
            thresholds.append(local_min)
        else:
            thresholds.append(np.inf)

    global_threshold = np.max(thresholds)  # Using mean of local minima as global threshold
    
    zero_crossings = np.where(histogram <= global_threshold)[0]
    start_row = 0
    lines = []

    for row in zero_crossings:
        if row - start_row > 10:  # Minimum line height
            line = preprocessed_img[start_row:row, :]
            lines.append(line)
        start_row = row

    return lines

def needs_reversal(line):
    # get baseline with the more white pixels
    baselinePos =  np.argmax(np.sum(line, axis=1))
    return baselinePos <= len(line) / 2



def text_rotation(image):
    kernel = np.ones((7, 7), np.uint8)
    image_canny = cv2.Canny(image,20,120)
    closing = cv2.morphologyEx(image_canny, cv2.MORPH_CLOSE, kernel)
    lines = cv2.HoughLinesP(closing, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)
        median_angle = np.median(angles)
    else:
        median_angle = 0 
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), median_angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    # rotated_image = cv2.resize(rotated_image, (width*2, height*2))
    
    # for reversal lines if needed
    lines = adaptive_line_segmentation(rotated_image)
    # for line in lines:
    #     cv2.imshow("line", line)
    #     cv2.waitKey(0)
    needRotation = False

    # take second line if possible
    if(len(lines)>1):
        needRotation = needs_reversal(lines[1])
    elif(len(lines)==1):
        needRotation = needs_reversal(lines[0])

    print(needRotation)

    height, width = rotated_image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 90, 1)
    if needRotation:
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))


   
    return rotated_image

def unblur_image(image):             
    return image


def preprocess(data):
    preprocessed_data=[]
    for image in data:
        # cv2.imshow("before preprocessing", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        noise_removed=salt_paper(image)
        # cv2.imshow("after salt and pepper preprocessing", noise_removed)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        binary=text_binary(noise_removed)
        # cv2.imshow("after binary preprocessing", binary)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        rotated=text_rotation(binary)
        # cv2.imshow("after rotaation in preprocessing", rotated)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        preprocessed_data.append(rotated)
 
    return preprocessed_data
    
    # image = cv2.imread(r"fonts-dataset\IBM Plex Sans Arabic\994.jpeg")
 
    # sharpened_image = unblur_image(image)
    # noise_removed=salt_paper( sharpened_image)
    # binary=text_binary(noise_removed)
    
    # cv2.imshow("Image", sharpened_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow("Image", rotated)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # for image in data:
    #     noise_removed=salt_paper(image)
    #     binary=text_binary(noise_removed)
    #     rotated=text_rotation(binary)
    #     cv2.imshow("Image", image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     cv2.imshow("Image", rotated)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()