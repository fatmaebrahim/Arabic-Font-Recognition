import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier


def train_data(data, datalabels):
    X_train=data
    y_train=datalabels
    random_forest = RandomForestClassifier()
    random_forest.fit(X_train, y_train)
    joblib.dump(random_forest, 'all.pkl')

def test_data(test, testlabels):
    X_test=test
    y_test=testlabels
    random_forest=joblib.load('all.pkl')
    y_pred = random_forest.predict(X_test)
    # print("ytest: ",y_test)
    # print("ypred: ",y_pred.tolist())      
    
    majority_class = np.bincount(y_pred).argmax()  
    # print("the majority of lines vote for: ",majority_class)
    return majority_class
