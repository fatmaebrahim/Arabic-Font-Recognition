import cv2
import os
from  preprocessing import preprocess
from modeltraining import lpq
import joblib
import pywt
import numpy as np
def read_data(folder_path,count):
    image_list = []
    for filename in os.listdir(folder_path):
        if count==0:
            break
        count-=1
        if filename.endswith((".jpeg")):  
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                image_list.append(image)

    return image_list


# folder_path = r"F:\LockD\CMP2025\Third_Year\Second_Term\Neural_Networks\Project\testing\3"
# image_list = read_data(folder_path)
# clf = joblib.load('Haar_LPQ.pkl')

# for image in image_list:
#     image=preprocess_image(image)
#     image=cv2.resize(image,(32,32))
#     feature_lpq=lpq(image)
#     feature_2d = feature_lpq.reshape(1, -1)
#     coeffs = pywt.dwt2(image, 'haar')  # Using Haar wavelet as an example
#     cA, (cH, cV, cD) = coeffs
#     feature_haar = np.concatenate((cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()))
#     features = np.concatenate((feature_haar, feature_lpq))
#     result = clf.predict(np.array(features).reshape(1, -1))
    
#     print (result)
