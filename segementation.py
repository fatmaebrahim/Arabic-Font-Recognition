import numpy as np
import cv2 as cv
from glob import glob
from scipy.ndimage import interpolation as inter
from PIL import Image as im
import cv2

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


def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score

def deskew(binary_img):

    
    ht, wd = binary_img.shape
    # _, binary_img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

    # pix = np.array(img.convert('1').getdata(), np.uint8)
    bin_img = (binary_img // 255.0)

    delta = 0.1
    limit = 3
    angles = np.arange(-limit, limit+delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(bin_img, angle)
        scores.append(score)

    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    # print('Best angle: {}'.formate(best_angle))

    # correct skew
    data = inter.rotate(bin_img, best_angle, reshape=False, order=0)
    img = im.fromarray((255 * data).astype("uint8"))

    # img.save('skew_corrected.png')
    pix = np.array(img)
    return pix

def save_image(img, folder, title):
    cv.imwrite(f'./{folder}/{title}.png', img)


def projection(gray_img, axis:str='horizontal'):
    """ Compute the horizontal or the vertical projection of a gray image """

    if axis == 'horizontal':
        projection_bins = np.sum(gray_img, 1).astype('int32')
    elif axis == 'vertical':
        projection_bins = np.sum(gray_img, 0).astype('int32')

    return projection_bins



def preprocess(image):

    # Maybe we end up using only gray level image.
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # gray_img = cv.bitwise_not(gray_img)

    binary_img = binary_otsus(gray_img, 0)
    # cv.imwrite('origin.png', gray_img)

    deskewed_img = deskew(binary_img)
    # deskewed_img = deskew(binary_img)
    # cv.imwrite('output.png', deskewed_img)

    binary_img = binary_otsus(deskewed_img, 0)
    # breakpoint()

    # Visualize

    # breakpoint()
    return binary_img


def projection_segmentation(clean_img, axis, cut=3):
    
    segments = []
    start = -1
    cnt = 0

    projection_bins = projection(clean_img, axis)
    for idx, projection_bin in enumerate(projection_bins):

        if projection_bin != 0:
            cnt = 0
        if projection_bin != 0 and start == -1:
            start = idx
        if projection_bin == 0 and start != -1:
            cnt += 1
            if cnt >= cut:
                if axis == 'horizontal':
                    line_img = clean_img[max(start-1, 0):idx, :]
                    # print(line_img.shape[0])
                    # print("///////////////////////////")
                    # Check if the line is too thin (e.g., a dot) and merge it with the previous line if needed
                    if line_img.shape[0] > 15:
                        segments.append(line_img)
                elif axis == 'vertical':
                    segments.append(clean_img[:, max(start-1, 0):idx])
                cnt = 0
                start = -1
    
    return segments



# Line Segmentation
#----------------------------------------------------------------------------------------
def line_horizontal_projection(image, line_height=50, cut=3):
    # Preprocess input image
    clean_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv2.imshow('Line', clean_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Segmentation
    lines = projection_segmentation(clean_img, axis='horizontal', cut=cut)

    # Resize lines to a fixed height
    # resized_lines = [cv2.resize(line, (line.shape[1], line_height)) for line in lines]


    for line in lines:
        cv2.imshow('Line', line)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return lines



if __name__ == "__main__":
    img = cv.imread(r'F:\LockD\CMP2025\Third_Year\Second_Term\Neural_Networks\Project\testing\0\953.jpeg')
    line_horizontal_projection(img)