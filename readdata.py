import cv2
import os
from  preprocessing import preprocess_image
from modeltraining import lpq
import joblib
def read_data(folder_path):
    image_list = []
    count=0
    for filename in os.listdir(folder_path):
        if count==100:
            break
        count+=1
        if filename.endswith((".jpeg")):  
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                image_list.append(image)
    return image_list

folder_path = r"F:\LockD\CMP2025\Third_Year\Second_Term\Neural_Networks\Project\testing\2"
image_list = read_data(folder_path)
clf = joblib.load('voting.pkl')

for image in image_list:
    image=preprocess_image(image)
    image=cv2.resize(image,(32,32))
    feature=lpq(image)
    feature_2d = feature.reshape(1, -1)
    result = clf.predict(feature_2d)
    
    print (result)
