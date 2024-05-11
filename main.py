import readdata
import preprocessing
import features
import modeltraining
import numpy as np
import time
import cv2
import modelbit
# mb = modelbit.login()

path_folder=r"F:\LockD\CMP2025\Third_Year\Second_Term\Neural_Networks\Project\fonts-dataset"
data_paths = [
    path_folder+r"\ibm",
    path_folder+r"\lem",
    path_folder+r"\mar",
    path_folder+r"\sha",
   
]

def TrainingModule():
    start_training_time = time.time()
    data=[]
    labels=[]
    datastart=300
    NumberOftrainingData=7
    readdata.read_data(data_paths[0],NumberOftrainingData,datastart,3,data,labels)
    readdata.read_data(data_paths[1],NumberOftrainingData,datastart,2, data,labels)
    readdata.read_data(data_paths[2],NumberOftrainingData,datastart,1,data,labels)
    readdata.read_data(data_paths[3],NumberOftrainingData,datastart,0, data,labels)
    preprocessed_data = preprocessing.preprocess(data)
    imagefeatures=[]
    linelabels=[]
    for j in range(len(preprocessed_data)):
        features.feature_extraction(labels[j],  preprocessed_data[j] , imagefeatures,linelabels)
    modeltraining.train_data(imagefeatures,linelabels)

    end_training_time = time.time()
    training_duration = end_training_time - start_training_time
    print("Training Time:", training_duration, "seconds")



def PredictionModule(image):
    image_list=[]
    image_list.append(image)
    start_testing_time = time.time()
    preprocessed_test=preprocessing.preprocess(image_list)
    allpredicted=[]
    for j in range(len(preprocessed_test)):
        testfeatures=[]
        testlinelabels=[]
        labels=[1]
        features.feature_extraction(labels[0],  preprocessed_test[j] , testfeatures,testlinelabels)
        if testfeatures!=[]:
            predictedlabel=modeltraining.test_data(testfeatures,testlinelabels)
            allpredicted.append(predictedlabel)
    end_testing_time = time.time()
    testing_duration = end_testing_time - start_testing_time
    print("Testing Time:", testing_duration, "seconds")
    return predictedlabel
def Compare(predicted_label,real_label):
    return predicted_label==real_label

def AccuracyModule(Result):    
    print("Accuracy: ",(Result.count(True)/len(Result)*100))
    
    
def TestingModule(testlabels,test):
    Result=[]
    for i in range(len(testlabels)):
        print("Real Label",testlabels[i])
        Predicted_label=PredictionModule(test[i])
        Result.append(Compare(Predicted_label,testlabels[i]))
    AccuracyModule(Result)    
    
test=[]
testlabels=[]
teststart=0
NumoftestData=50

def ReadTestData():
    
    readdata.read_all_data(data_paths[0],NumoftestData,3,test,testlabels)
    readdata.read_all_data(data_paths[1],NumoftestData,2,test,testlabels)
    readdata.read_all_data(data_paths[2],NumoftestData,1,test,testlabels)
    readdata.read_all_data(data_paths[3],NumoftestData,0,test,testlabels)

# image=cv2.imread(r"F:\LockD\CMP2025\Third_Year\Second_Term\Neural_Networks\NN_dataset\fonts-dataset\IBMPlexSansArabic\59.jpeg")
# print(image)
# predicted_label=PredictionModule(image)
# TestingModule(testlabels,test)
# print(Compare(predicted_label,3))
# print("Server Result")
# print(modelbit.get_inference(
#   workspace="eng-st-cu-edu",
#   deployment="test_image",
#   data=image
# ))
# print(test_image([image]))


ReadTestData()
TestingModule(testlabels,test)
    