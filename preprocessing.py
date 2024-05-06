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

def salt_paper(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised_image = cv2.medianBlur(image, 3)
    return denoised_image


def text_binary(image):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # np.set_printoptions(threshold=np.inf)
    # image_canny = cv2.Canny(grayscale_image,20,120)
    # return image_canny
    pixels = image.reshape(-1, 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 2
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.reshape(image.shape)
    counts = np.bincount(labels.flatten())
    foreground_cluster = np.argmax(counts)
    binary_image= np.where(labels == foreground_cluster, 0, 255).astype(np.uint8)
    return binary_image



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
    rotated_image = cv2.resize(rotated_image, (width*2, height*2))

   

   
    return rotated_image

def unblur_image(image):
    return image


def preprocess(data):
    preprocessed_data=[]
    for image in data:
        noise_removed=salt_paper(image)
        binary=text_binary(noise_removed)
        # rotated=text_rotation( binary)
        # cv2.imshow("Image", binary)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        preprocessed_data.append(binary)
 
        
        
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