import pickle

from sklearn.model_selection import train_test_split

from models.model_SVC import svc_linear
from src.models.model_DTC import dtc_model
from src.models.model_GNB import gnb_model
from src.models.model_KNN import knn_model
from src.models.model_RFC import rfc_model


def fit_models(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    accuracies = [0] * 5

    svc, accuracies[0] = svc_linear(X_train, X_test, y_train, y_test)
    gnb, accuracies[1] = gnb_model(X_train, X_test, y_train, y_test)
    knn, accuracies[2] = knn_model(X_train, X_test, y_train, y_test)
    rfc, accuracies[3] = rfc_model(X_train, X_test, y_train, y_test)
    dtc, accuracies[4] = dtc_model(X_train, X_test, y_train, y_test)

    print(accuracies)
    # SVC and KNN best accuracies!
    # KNN better after cleaning data
    # RFC 100% almost all time

    svc.fit(X, y)
    pickle.dump(svc, open("svc.mdl", 'wb'))

    gnb.fit(X, y)
    pickle.dump(gnb, open("gnb.mdl", 'wb'))

    knn.fit(X, y)
    pickle.dump(knn, open("knn.mdl", 'wb'))

    rfc.fit(X, y)
    pickle.dump(rfc, open("rfc.mdl", 'wb'))

    dtc.fit(X, y)
    pickle.dump(dtc, open("dtc.mdl", 'wb'))
