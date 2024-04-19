from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # Example model, replace with actual models
from scipy.signal import convolve2d
from sklearn import svm ,metrics
import numpy as np
import os
import cv2 
import joblib
from preprocessing import preprocess_image
    
def lpq(img,winSize=5):

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

# Create an SVM classifier
clf = svm.SVC()
training_data=[]
training_labels=[]
Number_of_images=100
def IBM_font_training():
    for root, dirs, files in os.walk(r"F:\LockD\CMP2025\Third_Year\Second_Term\Neural_Networks\NN_dataset\fonts-dataset\IBMPlexSansArabic"):
        i=0
        for file in files:
            if file.endswith('.jpeg') and i < Number_of_images:
                i=i+1
                path = os.path.join(root, file)
                print(path)
                image = cv2.imread(path)
                preprocessed_image = preprocess_image(image)
                image=cv2.resize(preprocessed_image,(32,32))
                feature=lpq(image)
                training_data.append(feature)
                training_labels.append(3)
   
def Lemonada_font_training():
    for root, dirs, files in os.walk(r"F:\LockD\CMP2025\Third_Year\Second_Term\Neural_Networks\NN_dataset\fonts-dataset\Lemonada"):
        i=0
        for file in files:
            if file.endswith('.jpeg') and i < Number_of_images:
                i=i+1
                path = os.path.join(root, file)
                print(path)
                image = cv2.imread(path)
                preprocessed_image = preprocess_image(image)
                image=cv2.resize(preprocessed_image,(32,32))
                feature=lpq(image)
                training_data.append(feature)
                training_labels.append(2)
   

def Marhey_font_training():
    for root, dirs, files in os.walk(r"F:\LockD\CMP2025\Third_Year\Second_Term\Neural_Networks\NN_dataset\fonts-dataset\Marhey"):
        i=0
        for file in files:
            if file.endswith('.jpeg') and i < Number_of_images:
                i=i+1
                path = os.path.join(root, file)
                print(path)
                image = cv2.imread(path)
                preprocessed_image = preprocess_image(image)
                image=cv2.resize(preprocessed_image,(32,32))
                feature=lpq(image)
                training_data.append(feature)
                training_labels.append(1)
   

def Scheherazade_New_font_training():
    for root, dirs, files in os.walk(r"F:\LockD\CMP2025\Third_Year\Second_Term\Neural_Networks\NN_dataset\fonts-dataset\ScheherazadeNew"):
        i=0
        for file in files:
            if file.endswith('.jpeg') and i < Number_of_images:
                i=i+1
                path = os.path.join(root, file)
                print(path)
                image = cv2.imread(path)
                preprocessed_image = preprocess_image(image)
                image=cv2.resize(preprocessed_image,(32,32))
                feature=lpq(image)
                training_data.append(feature)
                training_labels.append(0)
   


#training data for different fonts
# IBM_font_training()
# Lemonada_font_training()
# Marhey_font_training()
# Scheherazade_New_font_training()
# clf.fit(training_data, training_labels)

# joblib.dump(clf, 'training2.pkl')
