import cv2
import os

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

# folder_path = "fonts-dataset\IBM Plex Sans Arabic"
# images = read_images(folder_path)
# print("Number of images found:", len(images))


