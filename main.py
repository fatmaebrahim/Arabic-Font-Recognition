import readdata
import preprocessing
import features
import modeltraining
import numpy as np
import time


start_training_time = time.time()
data_paths = [
    r"D:\Fatma\2ndTerm_3rdYear\NN\Project\fonts-dataset\IBM Plex Sans Arabic",
    r"D:\Fatma\2ndTerm_3rdYear\NN\Project\fonts-dataset\Lemonada",
    r"D:\Fatma\2ndTerm_3rdYear\NN\Project\fonts-dataset\Marhey",
    r"D:\Fatma\2ndTerm_3rdYear\NN\Project\fonts-dataset\Scheherazade New"
]
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



start_testing_time = time.time()
test=[]
testlabels=[]
teststart=0
NumoftestData=2
readdata.read_data(data_paths[0],NumoftestData,teststart,3,test,testlabels)
readdata.read_data(data_paths[1],NumoftestData,teststart,2,test,testlabels)
readdata.read_data(data_paths[2],NumoftestData,teststart,1,test,testlabels)
readdata.read_data(data_paths[3],NumoftestData,teststart,0,test,testlabels)
preprocessed_test=preprocessing.preprocess(test)


allcorrect=[]
allpredicted=[]
for j in range(len(preprocessed_test)):
    testfeatures=[]
    testlinelabels=[]
    features.feature_extraction(testlabels[j],  preprocessed_test[j] , testfeatures,testlinelabels)
    if testfeatures!=[]:
        predictedlabel=modeltraining.test_data(testfeatures,testlinelabels)
        allcorrect.append(testlabels[j])
        allpredicted.append(predictedlabel)
        print("predicted label is:",predictedlabel,"correct label is:",testlabels[j])

end_testing_time = time.time()
testing_duration = end_testing_time - start_testing_time
print("Testing Time:", testing_duration, "seconds")
print(allcorrect)
print(allpredicted)
accuracy_percentage = (np.sum(np.asarray(allcorrect) == np.asarray(allpredicted)) / len(allpredicted)) * 100
print("total accuracy (%):", accuracy_percentage)