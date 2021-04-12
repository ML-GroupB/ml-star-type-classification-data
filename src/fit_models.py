import pickle

from sklearn.model_selection import train_test_split

from models.model_SVC import svc_linear
from src.models.model_GNB import gnb_model
from src.models.model_KNN import knn_model


def fit_models(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    accuracies = [0] * 3

    svc, accuracies[0] = svc_linear(X_train, X_test, y_train, y_test)
    knn, accuracies[2] = knn_model(X_train, X_test, y_train, y_test)
    gnb, accuracies[1] = gnb_model(X_train, X_test, y_train, y_test)

    print(accuracies)
    # SVC and KNN best accuracies!
    # KNN better after cleaning data

    svc.fit(X, y)
    pickle.dump(svc, open("svc.mdl", 'wb'))

    knn.fit(X, y)
    pickle.dump(knn, open("knn.mdl", 'wb'))

    gnb.fit(X, y)
    pickle.dump(gnb, open("gnb.mdl", 'wb'))
