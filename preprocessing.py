# Text alignment
# Text size/weight
# Image blur
# Brightness variation
# Text color
# Text rotation
# Salt and pepper noise
import numpy as np
import cv2
from skimage.feature import canny
from skimage.filters import sobel

def salt_paper(image):
    denoised_image = cv2.medianBlur(image, 3)
    return denoised_image


def text_binary(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    np.set_printoptions(threshold=np.inf)
    image_canny = cv2.Canny(grayscale_image,20,120)
    return image_canny
   


def text_rotation(image):
    lines = cv2.HoughLinesP(image, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)
    median_angle = np.median(angles)
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), median_angle, 0.5)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image


def preprocess(data):
    for image in data:
        noise_removed=salt_paper(image)
        binary=text_binary(noise_removed)
        rotated=text_rotation(binary)
        cv2.imshow("Image", rotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    






