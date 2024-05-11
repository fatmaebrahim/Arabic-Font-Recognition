import numpy as np
import cv2
import math
from scipy.signal import convolve2d



def lpq(img,winSize=3):

    STFTalpha=1/winSize  
    convmode='valid' 

    img=np.float64(img) 
    r=(winSize-1)/2 
    x=np.arange(-r,r+1)[np.newaxis] 

   
    w0=np.ones_like(x)
    w1=np.exp(-2*np.pi*x*STFTalpha*1j)
    w2=np.conj(w1)

    filterResp1=convolve2d(convolve2d(img,w0.T,convmode),w1,convmode)
    filterResp2=convolve2d(convolve2d(img,w1.T,convmode),w0,convmode)
    filterResp3=convolve2d(convolve2d(img,w1.T,convmode),w1,convmode)
    filterResp4=convolve2d(convolve2d(img,w1.T,convmode),w2,convmode)
    freqResp=np.dstack([filterResp1.real, filterResp1.imag,
                        filterResp2.real, filterResp2.imag,
                        filterResp3.real, filterResp3.imag,
                        filterResp4.real, filterResp4.imag])

    inds = np.arange(freqResp.shape[2])[np.newaxis,np.newaxis,:]
    LPQdesc=((freqResp>0)*(2**inds)).sum(2)
    LPQdesc=np.histogram(LPQdesc.flatten(),range(256))[0]
    LPQdesc=LPQdesc/LPQdesc.sum()
    return LPQdesc


def adaptive_line_segmentation(preprocessed_img, segment_size=100):
    histogram = np.sum(preprocessed_img, axis=1)
    n_segments = len(histogram) // segment_size
    thresholds = []
    for i in range(n_segments + 1):
        start = i * segment_size
        end = min((i + 1) * segment_size, len(histogram))
        local_hist = histogram[start:end]
        if len(local_hist) > 0:
            local_min = np.min(local_hist)
            thresholds.append(local_min)
        else:
            thresholds.append(np.inf)

    global_threshold = np.max(thresholds)  
    
    zero_crossings = np.where(histogram <= global_threshold)[0]
    start_row = 0
    lines = []

    for row in zero_crossings:
        if row - start_row > 10:  
            line = preprocessed_img[start_row:row, :]
            lines.append(line)
        start_row = row

    return lines

def angles(image):
    threshold=20
    min_line_length=20
    max_line_gap=10
    edges = cv2.Canny(image, 20, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    max_diff = 0
    degrees = []
    feature_angles=None
    if lines is not None:
        y1_values = lines[:, 0, 1]
        y2_values = lines[:, 0, 3]
        absolute_diff = np.abs(y1_values - y2_values)
        max_diff = np.max(absolute_diff)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < abs(y2 - y1):
                slope = (y1 - y2) / (x2 - x1 + 0.00001)
                angle_rad = math.atan(slope)  
                angle_deg = math.degrees(angle_rad)  
                if np.abs(angle_deg) > 75 and np.abs(y1 - y2) >max_diff-50:
                    degrees.append(np.abs(angle_deg))
       
        if degrees!=[]:
            feature_angles=np.mean(degrees)
        return feature_angles
    
def feature_extraction(font, whole_image,data,labels):

    lines_image=adaptive_line_segmentation(whole_image)
    for image in lines_image:
        feature_angles=angles(image)
        feature_lpq=lpq(image)
        if feature_angles is None:
            features=np.concatenate((feature_lpq,np.array([0.0])))
        else:
            features=np.concatenate((feature_lpq,np.array([feature_angles])))
        data.append(features)
        labels.append(font)

    
