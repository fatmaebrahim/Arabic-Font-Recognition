import cv2
import os

def read_data(folder_path, count,start, font, data, labels):
    for i in range(count):
        filename = f"{start+i}.jpeg"
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        if image is not None:
            data.append(image)
            labels.append(font)



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
