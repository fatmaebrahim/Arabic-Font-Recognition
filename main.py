import readdata
import preprocessing
import features
import modeltraining
# import performance
# from sklearn.model_selection import train_test_split

# Load and preprocess data

data=[]
anglefeature=[]
data.append( readdata.read_data("fonts-dataset\IBM Plex Sans Arabic",500))
data.append( readdata.read_data("fonts-dataset\Lemonada",500))
data.append( readdata.read_data("fonts-dataset\Marhey",500))
data.append( readdata.read_data("fonts-dataset\Scheherazade New",500))

for i in range(len(data)):
    preprocessed_data = preprocessing.preprocess(data[i])
    features.detect_lines(i,  preprocessed_data , anglefeature)



test=[]
test.append( readdata.read_data("fonts-dataset\ibm",20))
test.append( readdata.read_data("fonts-dataset\lem",20))
test.append( readdata.read_data("fonts-dataset\mar",20))
test.append( readdata.read_data("fonts-dataset\sha",20))
testdata=[]
for i in range(len(test)):
    preprocessed_data = preprocessing.preprocess(test[i])
    features.detect_lines(i,  preprocessed_data ,testdata)

predicted_labels = modeltraining.classify_data(anglefeature,testdata)