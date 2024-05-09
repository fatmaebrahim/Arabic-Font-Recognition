import readdata
import preprocessing
import features
import modeltraining
import numpy as np


data=[]
labels=[]
NumberOftrainingData=50
readdata.read_data(r"D:\Fatma\2ndTerm_3rdYear\NN\Project\fonts-dataset\IBM Plex Sans Arabic",NumberOftrainingData,0,data,labels)
readdata.read_data(r"D:\Fatma\2ndTerm_3rdYear\NN\Project\fonts-dataset\Lemonada",NumberOftrainingData,1, data,labels)
readdata.read_data(r"D:\Fatma\2ndTerm_3rdYear\NN\Project\fonts-dataset\Marhey",NumberOftrainingData,2,data,labels)
readdata.read_data(r"D:\Fatma\2ndTerm_3rdYear\NN\Project\fonts-dataset\Scheherazade New",NumberOftrainingData,3, data,labels)
preprocessed_data = preprocessing.preprocess(data)
imagefeatures=[]
linelabels=[]
for j in range(len(preprocessed_data)):
    features.feature_extraction(labels[j],  preprocessed_data[j] , imagefeatures,linelabels)

test=[]
testlabels=[]
NumoftestData=10
readdata.read_data(r"D:\Fatma\2ndTerm_3rdYear\NN\Project\fonts-dataset\ibm",NumoftestData,0,test,testlabels)
readdata.read_data(r"D:\Fatma\2ndTerm_3rdYear\NN\Project\fonts-dataset\lem",NumoftestData,1,test,testlabels)
readdata.read_data(r"D:\Fatma\2ndTerm_3rdYear\NN\Project\fonts-dataset\mar",NumoftestData,2,test,testlabels)
readdata.read_data(r"D:\Fatma\2ndTerm_3rdYear\NN\Project\fonts-dataset\sha",NumoftestData,3,test,testlabels)
preprocessed_test=preprocessing.preprocess(test)
allcorrect=[]
allpredicted=[]
for j in range(len(preprocessed_test)):
    testfeatures=[]
    testlinelabels=[]
    features.feature_extraction(testlabels[j],  preprocessed_test[j] , testfeatures,testlinelabels)
    
    predictedlabel=modeltraining.classify_data(imagefeatures,linelabels,testfeatures,testlinelabels)
    allcorrect.append(testlabels[j])
    allpredicted.append(predictedlabel)
    print("predicted label is:",predictedlabel,"correct label is:",testlabels[j])


print(allcorrect)
print(allpredicted)
accuracy_percentage = (np.sum(np.asarray(allcorrect) == np.asarray(allpredicted)) / len(allpredicted)) * 100
print("total accuracy (%):", accuracy_percentage)