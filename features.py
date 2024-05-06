import numpy as np
import cv2
import csv
import math
from scipy.signal import convolve2d
import pywt

def lpq(img,winSize=3):

    #using nh mode
    
    STFTalpha=1/winSize  # alpha in STFT approaches (for Gaussian derivative alpha=1)
    convmode='valid' # Compute descriptor responses only on part that have full neigborhood. Use 'same' if all pixels are included (extrapolates np.image with zeros).

    img=np.float64(img) # Convert np.image to double
    r=(winSize-1)/2 # Get radius from window size
    x=np.arange(-r,r+1)[np.newaxis] # Form spatial coordinates in window

    #  STFT uniform window Basic STFT filters
    w0=np.ones_like(x)
    w1=np.exp(-2*np.pi*x*STFTalpha*1j)
    w2=np.conj(w1)

    ## Run filters to compute the frequency response in the four points. Store np.real and np.imaginary parts separately
    # Run first filter
    filterResp1=convolve2d(convolve2d(img,w0.T,convmode),w1,convmode)
    filterResp2=convolve2d(convolve2d(img,w1.T,convmode),w0,convmode)
    filterResp3=convolve2d(convolve2d(img,w1.T,convmode),w1,convmode)
    filterResp4=convolve2d(convolve2d(img,w1.T,convmode),w2,convmode)

    # Initilize frequency domain matrix for four frequency coordinates (np.real and np.imaginary parts for each frequency).
    freqResp=np.dstack([filterResp1.real, filterResp1.imag,
                        filterResp2.real, filterResp2.imag,
                        filterResp3.real, filterResp3.imag,
                        filterResp4.real, filterResp4.imag])

    ## Perform quantization and compute LPQ codewords
    inds = np.arange(freqResp.shape[2])[np.newaxis,np.newaxis,:]
    LPQdesc=((freqResp>0)*(2**inds)).sum(2)

    ## Histogram if needed
    LPQdesc=np.histogram(LPQdesc.flatten(),range(256))[0]

    ## Normalize histogram if needed
    LPQdesc=LPQdesc/LPQdesc.sum()

    #print(LPQdesc)
    return LPQdesc

def feature_extraction(font, images,data,labels):
    threshold=20
    min_line_length=20
    max_line_gap=10
    for image in images:
        
        edges = cv2.Canny(image, 20, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
        max_diff = 0
        # if lines is not None:
            # for line in lines:
            #     x1, y1, x2, y2 = line[0]
            #     max_diff = max(max_diff, np.abs(y1 - y2))
        degrees = []
        if lines is not None:
            y1_values = lines[:, 0, 1]
            y2_values = lines[:, 0, 3]
            absolute_diff = np.abs(y1_values - y2_values)
            max_diff = np.max(absolute_diff)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < abs(y2 - y1):
                    slope = (y1 - y2) / (x2 - x1 + 0.00001)
                    angle_rad = math.atan(slope)  # Calculate angle in radians
                    angle_deg = math.degrees(angle_rad)  # Convert angle to degrees
                    if np.abs(angle_deg) > 75 and np.abs(y1 - y2) >max_diff-50:
                        degrees.append(np.abs(angle_deg))
                        # cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.imshow("Image", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            feature_angles=None
            if degrees!=[]:
                feature_angles=np.mean(degrees)
                # print([np.mean(degrees),font])
        coeffs = pywt.dwt2(image, 'haar')  # Using Haar wavelet as an example
        cA, (cH, cV, cD) = coeffs
        feature_haar = np.concatenate((cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()))
        
        image=cv2.resize(image,(32,32))
        feature_lpq=lpq(image)
        
        
        if feature_angles is None:
            features=np.concatenate((feature_lpq, feature_haar,np.array([0.0])))
        else:
            features=np.concatenate((feature_lpq, feature_haar,np.array([feature_angles])))
        data.append(features)
        labels.append(font)
        print("////////////////////labels")
        print(labels)
    
