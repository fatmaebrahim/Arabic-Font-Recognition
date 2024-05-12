import readdata
import preprocessing
import features
import modeltraining
import time
import cv2
# mb = modelbit.login()

path_folder=r"D:\Fatma\2ndTerm_3rdYear\NN\Project\fonts-dataset"
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


test_timing=[]
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
    test_timing.append(round(testing_duration,3))
    return predictedlabel
def Compare(predicted_label,real_label):
    return predicted_label==real_label

def AccuracyModule(Result):    
    print("Accuracy: ",(Result.count(True)/len(Result)*100))
    
    
def TestingModule(testlabels,test):
    Result=[]
    Predicted_labels=[]
    for i in range(len(testlabels)):
        print("Real Label",testlabels[i])
        Predicted_label=PredictionModule(test[i])
        Predicted_labels.append(Predicted_label)
        Result.append(Compare(Predicted_label,testlabels[i]))
    
    with open('results.txt', 'w') as file:
        for item in Predicted_labels:
            file.write(str(item) + '\n')  
    with open('time.txt', 'w') as file:
        for item in test_timing:
            file.write(str(item) + '\n')  
            
    AccuracyModule(Result)    
    
test=[]
testlabels=[]
teststart=0
NumoftestData=5

def ReadTestData(): 
    readdata.read_all_data(data_paths[0],NumoftestData,3,test,testlabels)
    readdata.read_all_data(data_paths[1],NumoftestData,2,test,testlabels)
    readdata.read_all_data(data_paths[2],NumoftestData,1,test,testlabels)
    readdata.read_all_data(data_paths[3],NumoftestData,0,test,testlabels)


ReadTestData()
TestingModule(testlabels,test)
    