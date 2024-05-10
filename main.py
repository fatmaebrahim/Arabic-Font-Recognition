import readdata
import preprocessing
import features
import modeltraining
import numpy as np


data=[]
labels=[]
NumberOftrainingData=200
training_path_folder=r"F:\LockD\CMP2025\Third_Year\Second_Term\Neural_Networks\Project\fonts-dataset"
readdata.read_data(training_path_folder+r"\IBM",NumberOftrainingData,3,data,labels)
readdata.read_data(training_path_folder+r"\Lemonada",NumberOftrainingData,2, data,labels)
readdata.read_data(training_path_folder+r"\Marhey",NumberOftrainingData,1,data,labels)
readdata.read_data(training_path_folder+r"\ScheherazadeNew",NumberOftrainingData,0, data,labels)
preprocessed_data = preprocessing.preprocess(data)
imagefeatures=[]
linelabels=[]
for j in range(len(preprocessed_data)):
    features.feature_extraction(labels[j],  preprocessed_data[j] , imagefeatures,linelabels)

test=[]
testlabels=[]
NumoftestData=10
test_path_folder=r"F:\LockD\CMP2025\Third_Year\Second_Term\Neural_Networks\Project\testing"
readdata.read_data(test_path_folder+r"\0",NumoftestData,0,test,testlabels)
readdata.read_data(test_path_folder+r"\1",NumoftestData,1,test,testlabels)
readdata.read_data(test_path_folder+r"\2",NumoftestData,2,test,testlabels)
readdata.read_data(test_path_folder+r"\3" ,NumoftestData,3,test,testlabels)
preprocessed_test=preprocessing.preprocess(test)
allcorrect=[]
allpredicted=[]
for j in range(len(preprocessed_test)):
    testfeatures=[]
    testlinelabels=[]
    features.feature_extraction(testlabels[j],  preprocessed_test[j] , testfeatures,testlinelabels)
    print(j)
    
    predictedlabel=modeltraining.classify_data(imagefeatures,linelabels,testfeatures,testlinelabels)
    predictedlabel=modeltraining.test_data(testfeatures,testlinelabels)
    allcorrect.append(testlabels[j])
    allpredicted.append(predictedlabel)
    print("predicted label is:",predictedlabel,"correct label is:",testlabels[j])


print(allcorrect)
print(allpredicted)
accuracy_percentage = (np.sum(np.asarray(allcorrect) == np.asarray(allpredicted)) / len(allpredicted)) * 100
print("total accuracy (%):", accuracy_percentage)