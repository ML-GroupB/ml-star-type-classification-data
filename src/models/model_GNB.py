from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


# 70-80% is usual but sometimes it gets even more (or lower)
def gnb_model(X_train, X_test, y_train, y_test):
    gnb = GaussianNB().fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return gnb, accuracy
