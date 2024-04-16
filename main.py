import readdata
import preprocessing
# import features
# import modeltraining
# import performance
# from sklearn.model_selection import train_test_split

# Load and preprocess data
folder_path = "fonts-dataset\IBM Plex Sans Arabic"
data = readdata.read_data(folder_path)
preprocessed_data = preprocessing.preprocess(data)


# # Extract and select features
# features = features.extract_features(filled_data)
# selected_features = features.select_features(features, target)

# # Split data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(selected_features, target, test_size=0.2)

# # Select and train modeltraining
# selected_model = modeltraining.select_model()
# trained_model = modeltraining.train_model(selected_model, X_train, y_train)

# # Evaluate modeltraining performance
# accuracy, confusion_matrix, report = performance.evaluate_performance(trained_model, X_test, y_test)

# print("Accuracy:", accuracy)
# print("Confusion Matrix:", confusion_matrix)
# print("Classification Report:", report)
