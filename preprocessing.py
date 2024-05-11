import numpy as np
import cv2
import cv2 as cv

def salt_paper(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised_image = cv2.medianBlur(image, 3)
    return denoised_image


def text_binary(image, filter:int=1):
    if len(image.shape) == 3:
        gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray_img = image

    if filter != 0:
        blur = cv.GaussianBlur(gray_img, (3,3), 0)
        binary_img = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
    else:
        binary_img = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
  
    white_count = cv.countNonZero(binary_img)
    black_count = len(binary_img) * len(binary_img[0]) - white_count
    if(white_count > black_count):
        binary_img = 255 - binary_img
    blur = cv.GaussianBlur(binary_img, (3,3), 0)
    return blur


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
   
    return rotated_image




def preprocess(data):
    preprocessed_data=[]
    for image in data:
        noise_removed=salt_paper(image)
        binary_filter=text_binary(noise_removed)
        rotated=text_rotation(binary_filter)
        preprocessed_data.append(rotated)
 
    return preprocessed_data
    
    # image = cv2.imread(r"fonts-dataset\IBM Plex Sans Arabic\994.jpeg")

  