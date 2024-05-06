import readdata
import preprocessing
import features
import modeltraining
# import performance
# from sklearn.model_selection import train_test_split

# Load and preprocess data

data=[]
anglefeature=[]
labels=[]
NumberOftrainingData=20
data.append( readdata.read_data(r"F:\LockD\CMP2025\Third_Year\Second_Term\Neural_Networks\Project\training_set\IBM",NumberOftrainingData))
data.append( readdata.read_data(r"F:\LockD\CMP2025\Third_Year\Second_Term\Neural_Networks\Project\training_set\Lemonda",NumberOftrainingData))
data.append( readdata.read_data(r"F:\LockD\CMP2025\Third_Year\Second_Term\Neural_Networks\Project\training_set\Marhey",NumberOftrainingData))
data.append( readdata.read_data(r"F:\LockD\CMP2025\Third_Year\Second_Term\Neural_Networks\Project\training_set\ScheherazadeNew",NumberOftrainingData))

for i in range(len(data)):
    preprocessed_data = preprocessing.preprocess(data[i])
    features.feature_extraction(i,  preprocessed_data , anglefeature,labels)
print("////////////////////////////////////////////////Angle Features")
print(anglefeature)


predicted_labels = modeltraining.train_data(anglefeature,labels)

test=[]
NumoftestData=1
labels=[]
test.append( readdata.read_data(r"F:\LockD\CMP2025\Third_Year\Second_Term\Neural_Networks\Project\testing\0",NumoftestData))
test.append( readdata.read_data(r"F:\LockD\CMP2025\Third_Year\Second_Term\Neural_Networks\Project\testing\1",NumoftestData))
test.append( readdata.read_data(r"F:\LockD\CMP2025\Third_Year\Second_Term\Neural_Networks\Project\testing\2",NumoftestData))
test.append( readdata.read_data(r"F:\LockD\CMP2025\Third_Year\Second_Term\Neural_Networks\Project\testing\3",NumoftestData))
testdata=[]
for i in range(len(test)):
    preprocessed_data = preprocessing.preprocess(test[i])
    features.feature_extraction(i,  preprocessed_data ,testdata,labels)

modeltraining.test_data(testdata,labels)