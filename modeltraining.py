from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # Example model, replace with actual models
from scipy.signal import convolve2d
from sklearn import svm ,metrics
import numpy as np
import os
import cv2 
import joblib
from preprocessing import preprocess
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
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

# Create an SVM classifier
# clf = svm.SVC()
training_data=[]
training_labels=[]
Number_of_images=500
def IBM_font_training():
    for root, dirs, files in os.walk(r"F:\LockD\CMP2025\Third_Year\Second_Term\Neural_Networks\NN_dataset\fonts-dataset\IBMPlexSansArabic"):
        i=0
        for file in files:
            if file.endswith('.jpeg') and i < Number_of_images:
                features=[]
                i=i+1
                path = os.path.join(root, file)
                print(path)
                image = cv2.imread(path)
                preprocessed_image = preprocess_image(image)
                image=cv2.resize(preprocessed_image,(32,32))
                coeffs = pywt.dwt2(image, 'haar') 
                cA, (cH, cV, cD) = coeffs
                feature_haar = np.concatenate((cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()))
                feature_lpq=lpq(image)
                features = np.concatenate((feature_haar, feature_lpq))
                training_data.append(np.array(features))
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
                coeffs = pywt.dwt2(image, 'haar')  # Using Haar wavelet as an example
                cA, (cH, cV, cD) = coeffs
                feature_haar = np.concatenate((cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()))
                feature_lpq=lpq(image)
                features = np.concatenate((feature_haar, feature_lpq))
                training_data.append(np.array(features))
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
                coeffs = pywt.dwt2(image, 'haar')  # Using Haar wavelet as an example
                cA, (cH, cV, cD) = coeffs
                feature_haar = np.concatenate((cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()))
                feature_lpq=lpq(image)
                features = np.concatenate((feature_haar, feature_lpq))
                training_data.append(np.array(features))
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
                preprocessed_image = preprocess(image)
                image=cv2.resize(preprocessed_image,(32,32))
                coeffs = pywt.dwt2(image, 'haar')  # Using Haar wavelet as an example
                cA, (cH, cV, cD) = coeffs
                feature_haar = np.concatenate((cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()))
                feature_lpq=lpq(image)
                features = np.concatenate((feature_haar, feature_lpq))
                training_data.append(np.array(features))
                training_labels.append(0)
   

def train_data(data,labels):
    # print("///////////////data0")
    # print(data[0])
    # print("///////////////X_Train1")
    # print(data[1])
    # print("///////////////X_Train2")
    # print(data[2])
    # print("///////////////X_Train3")
    # print(data[3])
    X_train = data
    y_train = labels
    print("////////////////////data length")
    print(len(X_train))
    print("////////////////////labels length")
    print(len(y_train))
    
    # print("///////////////X_Train0")
    # print(X_train[0])
    # print("///////////////X_Train1")
    # print(X_train[1])
    # print("///////////////X_Train2")
    # print(X_train[2])
    # print("///////////////X_Train3")
    # print(X_train[3])
    # print("///////////////Train")
    # print(y_train)
    # Initialize Random Forest Classifier with default settings
    random_forest = RandomForestClassifier()
    
    # Train the model
    random_forest.fit(X_train, y_train)
    joblib.dump(random_forest, 'Haar_angles_LPQ.pkl')

    
def test_data( testdata,labels):
    random_forest=joblib.load('Haar_angles_LPQ.pkl')
    
    X_test = testdata
    y_test = labels
    print("///////////////Real")
    print(y_test)
    print("///////////////predict")

    # Make predictions
    y_pred = random_forest.predict(X_test)
    print(y_pred)

    # Count the correct predictions
    num_correct = np.sum(y_pred == y_test)
    
    # Calculate accuracy
    accuracy_percentage = (num_correct / len(y_test)) * 100
    print("Accuracy (%):", accuracy_percentage)

    return accuracy_percentage
